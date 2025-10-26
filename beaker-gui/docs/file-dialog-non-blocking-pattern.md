# File Dialog Non-Blocking Pattern

## Problem

When integrating native file dialogs into egui applications using the `rfd` crate, a common mistake is to use `pollster::block_on()` with `AsyncFileDialog` on the main UI thread. This causes the entire UI to freeze while waiting for the user to select a file.

### The Problematic Code

```rust
fn open_file_dialog() -> Option<PathBuf> {
    let future = rfd::AsyncFileDialog::new()
        .add_filter("Images", &["jpg", "jpeg", "png"])
        .pick_file();
    let result = pollster::block_on(future).map(|f| f.path().to_path_buf());
    result
}
```

**Why this freezes:**
1. This code is called from within egui's `update()` method, which runs on the main UI thread
2. `pollster::block_on()` blocks the current thread waiting for the future to complete
3. While blocked, egui cannot process events or render frames
4. The UI appears frozen until the file dialog closes

## Solution: Threading Pattern

The idiomatic solution in Rust for egui applications is to use a **threading pattern with shared state** using `Arc<Mutex<>>`.

### Implementation

#### 1. Add Shared State to Your Struct

```rust
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

pub struct WelcomeView {
    // Other fields...

    /// Pending file dialog result
    /// - None: No dialog currently open
    /// - Some(None): Dialog was cancelled
    /// - Some(Some(path)): File was selected
    pending_file_dialog: Arc<Mutex<Option<Option<PathBuf>>>>,
    pending_folder_dialog: Arc<Mutex<Option<Option<PathBuf>>>>,
}
```

#### 2. Initialize in Constructor

```rust
impl WelcomeView {
    pub fn new() -> Self {
        Self {
            // Other fields...
            pending_file_dialog: Arc::new(Mutex::new(None)),
            pending_folder_dialog: Arc::new(Mutex::new(None)),
        }
    }
}
```

#### 3. Spawn Dialog in Background Thread

Use the **synchronous** `FileDialog` (not `AsyncFileDialog`) in a spawned thread:

```rust
fn spawn_file_dialog(&self) {
    let dialog_result = Arc::clone(&self.pending_file_dialog);
    std::thread::spawn(move || {
        eprintln!("[WelcomeView] File dialog thread started");
        let path = rfd::FileDialog::new()
            .add_filter("Images", &["jpg", "jpeg", "png"])
            .add_filter("Beaker metadata", &["toml"])
            .pick_file();
        eprintln!("[WelcomeView] File dialog result: {:?}", path);
        *dialog_result.lock().unwrap() = Some(path);
    });
}

fn spawn_folder_dialog(&self) {
    let dialog_result = Arc::clone(&self.pending_folder_dialog);
    std::thread::spawn(move || {
        eprintln!("[WelcomeView] Folder dialog thread started");
        let path = rfd::FileDialog::new().pick_folder();
        eprintln!("[WelcomeView] Folder dialog result: {:?}", path);
        *dialog_result.lock().unwrap() = Some(path);
    });
}
```

#### 4. Check for Results in Update Loop (Non-Blocking)

At the start of your `show()` or `update()` method:

```rust
pub fn show(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> WelcomeAction {
    let mut action = WelcomeAction::None;

    // Check for completed file dialogs (non-blocking)
    if let Ok(mut result) = self.pending_file_dialog.try_lock() {
        if let Some(path_option) = result.take() {
            if let Some(path) = path_option {
                eprintln!("[WelcomeView] File dialog completed with path: {:?}", path);
                action = WelcomeAction::OpenImage(path);
            } else {
                eprintln!("[WelcomeView] File dialog cancelled");
            }
        }
    }

    // Check for completed folder dialogs (non-blocking)
    if let Ok(mut result) = self.pending_folder_dialog.try_lock() {
        if let Some(path_option) = result.take() {
            if let Some(path) = path_option {
                eprintln!("[WelcomeView] Folder dialog completed with path: {:?}", path);
                action = WelcomeAction::OpenFolder(path);
            } else {
                eprintln!("[WelcomeView] Folder dialog cancelled");
            }
        }
    }

    // ... rest of UI code
}
```

#### 5. Call from UI Events

```rust
if ui.button("Open Image").clicked() {
    self.spawn_file_dialog();
}

if ui.button("Open Folder").clicked() {
    self.spawn_folder_dialog();
}
```

## Key Benefits

1. **Non-Blocking**: The UI remains responsive while the file dialog is open
2. **Cross-Platform**: Works reliably on macOS, Linux, and Windows
3. **Simple**: Uses standard library threading primitives (`Arc`, `Mutex`, `thread::spawn`)
4. **Idiomatic**: Follows Rust best practices for shared mutable state

## Platform Considerations

### macOS
- The `rfd` crate recommends spawning dialogs on the main thread for best performance
- However, for windowed applications (like egui with winit/eframe), spawning from any thread works
- The synchronous `FileDialog` API works reliably across all platforms when called from a spawned thread

### Why Synchronous Instead of Async?
- The synchronous `FileDialog::new().pick_file()` is simpler to use in this pattern
- No need for async executors or `pollster::block_on`
- The dialog runs in its own thread, so it doesn't block the UI regardless
- On macOS, `AsyncFileDialog` requires an `NSApplication` instance which may not be available in all contexts

## References

- [egui Discussion #5621: How to avoid RFD file dialogs hanging egui?](https://github.com/emilk/egui/discussions/5621)
- [egui PR #5697: Update file dialog example to be non-blocking](https://github.com/emilk/egui/pull/5697)
- [rfd crate documentation](https://docs.rs/rfd/)

## Migration Checklist

If you have existing code using `pollster::block_on` with `AsyncFileDialog`:

- [ ] Add `Arc<Mutex<Option<Option<PathBuf>>>>` fields to your struct
- [ ] Initialize them in the constructor
- [ ] Replace `open_file_dialog()` calls with `spawn_file_dialog()`
- [ ] Add non-blocking result checking at the start of your update loop using `try_lock()`
- [ ] Remove `pollster` dependency from `Cargo.toml`
- [ ] Switch from `AsyncFileDialog` to synchronous `FileDialog`
- [ ] Test on your target platforms (especially macOS if applicable)
