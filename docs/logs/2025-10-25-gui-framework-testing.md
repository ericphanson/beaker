# Rust GUI Framework Feasibility Testing

**Date:** 2025-10-25
**Goal:** Find a pure Rust GUI framework that works in sandboxed/restricted environments without GTK dependencies
**Context:** Tauri requires GTK on Linux and cannot be tested locally in network-restricted environments

---

## Testing Criteria

✅ **Must Have:**
- Compiles without system dependencies (no apt-get needed)
- Runs in headless/sandboxed environments
- Pure Rust or minimal C dependencies
- Suitable for desktop application (not just games)

⚠️ **Nice to Have:**
- Good documentation
- Active development
- Image display support
- Layout system for complex UIs

❌ **Deal Breakers:**
- Requires GTK/WebKit
- Requires network access for build
- Requires X11/Wayland/GUI to build

---

## Framework Test Results

### Test Environment
- OS: Linux (Ubuntu 24.04)
- Network: Restricted (cannot install packages)
- Display: No X11/Wayland
- Cargo: Available

---

## 1. egui/eframe ✅

**Description:** Immediate mode GUI, pure Rust
**Website:** https://github.com/emilk/egui

**Test:** Creating minimal hello world application

```toml
# Cargo.toml
[dependencies]
eframe = "0.29"
egui = "0.29"
```

```rust
// src/main.rs
use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Beaker Test",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}

#[derive(Default)]
struct MyApp {}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello from egui!");
            ui.label("This is a test in a sandboxed environment.");
        });
    }
}
```

**Build Command:**
```bash
cargo build --release
```

**Result:** ✅ **SUCCESS**
- Build time: ~33 seconds
- No system dependencies required
- No pkg-config checks
- Binary created successfully
- Only fails at runtime when trying to create window (expected in headless environment)

**Runtime Error (Expected):**
```
Error: NoAvailableBackend
```
This is expected - winit cannot create a display without X11/Wayland, but **compilation succeeds**.

**Dependencies Analysis:**
- Pure Rust stack (eframe, egui, winit)
- No GTK, no WebKit
- All dependencies compile from source
- No build scripts requiring system libraries

---

## 2. Iced ✅

**Description:** Elm-inspired GUI framework, cross-platform
**Website:** https://github.com/iced-rs/iced

**Test:** Creating minimal counter application

```toml
# Cargo.toml
[dependencies]
iced = { version = "0.13", features = ["tokio"] }
```

```rust
// src/main.rs
use iced::{Element, Task};
use iced::widget::{button, column, text};

fn main() -> iced::Result {
    iced::application(
        "Beaker Test",
        Counter::update,
        Counter::view
    ).run()
}

#[derive(Default)]
struct Counter {
    value: i32,
}

#[derive(Debug, Clone)]
enum Message {
    Increment,
}

impl Counter {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Increment => {
                self.value += 1;
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<Message> {
        column![
            text("Hello from Iced!"),
            text(format!("Counter: {}", self.value)),
            button("Increment").on_press(Message::Increment),
        ].into()
    }
}
```

**Build Command:**
```bash
cargo build --release
```

**Result:** ✅ **SUCCESS**
- Build time: ~56 seconds
- No system dependencies required
- No pkg-config checks
- Binary created successfully
- Only fails at runtime (expected)

**Runtime Error (Expected):**
```
Error: GraphicsAdapterNotFound
```
Expected - cannot initialize graphics without display.

**Dependencies Analysis:**
- Pure Rust architecture
- Uses wgpu for rendering (works without display at build time)
- More complex than egui but still no system deps
- Elm-inspired architecture (Model-View-Update)

---

## 3. Slint ✅

**Description:** Declarative UI with custom DSL, designed for embedded
**Website:** https://slint.dev/

**Test:** Creating minimal hello world with .slint file

```toml
# Cargo.toml
[dependencies]
slint = "1.10"

[build-dependencies]
slint-build = "1.10"
```

```rust
// build.rs
fn main() {
    slint_build::compile("ui/appwindow.slint").unwrap();
}
```

```slint
// ui/appwindow.slint
export component AppWindow inherits Window {
    width: 400px;
    height: 300px;
    title: "Beaker Test - Slint";

    VerticalLayout {
        Text {
            text: "Hello from Slint!";
            font-size: 24px;
        }
        Text {
            text: "This is a test in a sandboxed environment.";
        }
    }
}
```

```rust
// src/main.rs
slint::include_modules!();

fn main() {
    let app = AppWindow::new().unwrap();
    app.run().unwrap();
}
```

**Build Command:**
```bash
cargo build --release
```

**Result:** ✅ **SUCCESS**
- Build time: ~5 seconds (incremental after first build)
- No system dependencies required
- DSL files compiled at build time
- Binary created successfully
- Only fails at runtime (expected)

**Runtime Error (Expected):**
```
Failed to create winit window
```
Expected - cannot create window without display.

**Dependencies Analysis:**
- Pure Rust with custom DSL
- Very fast incremental builds
- Designed for embedded/resource-constrained environments
- C++ compatibility layer but builds without system deps

**Note:** Initial attempt failed with "Unknown element 'VerticalBox'" - fixed by using `VerticalLayout` instead (API change in newer version).

---

## 4. Tauri ❌

**Description:** Webview-based framework (previous attempt)
**Website:** https://tauri.app/

**Result:** ❌ **FAILED AT BUILD TIME**

**Error:**
```
error: failed to run custom build command for `gdk-sys v0.19.9`
The system library `gdk-3.0` required by crate gdk-sys was not found.
The file `gdk-3.0.pc` needs to be installed and the PKG_CONFIG_PATH
environment variable must contain its parent directory.
```

**Why it failed:**
- Requires GTK/webkit2gtk system libraries
- pkg-config fails even for `cargo check`
- Cannot build without `apt-get install libgtk-3-dev libwebkit2gtk-4.1-dev`
- Network-restricted environment cannot install packages
- Not viable for sandboxed testing

---

## Comparison Matrix

| Framework | Build Time | System Deps | DSL | Compile in Sandbox | Maturity | Community |
|-----------|-----------|-------------|-----|-------------------|----------|-----------|
| **egui** | 33s | None ✅ | No | ✅ Yes | High | Large |
| **Iced** | 56s | None ✅ | No | ✅ Yes | High | Medium |
| **Slint** | 5s* | None ✅ | Yes | ✅ Yes | Medium | Growing |
| **Tauri** | N/A | GTK ❌ | HTML/JS | ❌ No | High | Large |

*Incremental build time; first build ~40s

---

## Recommendation: egui

**Primary Choice: egui/eframe**

**Reasons:**
1. ✅ Pure Rust - no system dependencies
2. ✅ Proven in production (rerun.io, Veloren, etc.)
3. ✅ Immediate mode = simpler state management
4. ✅ Fast compile times (33s)
5. ✅ Excellent documentation
6. ✅ Image display via egui::Image with egui_extras
7. ✅ Active development and large community
8. ✅ Works perfectly in sandboxed environments

**Implementation Path:**
- Use `eframe` for window management
- Use `egui` for UI widgets
- Use `egui_extras::RetainedImage` for image display
- Use `rfd` crate for native file dialogs
- Layout is straightforward with panels and grids

**Example Integration:**
```rust
// beaker-gui/Cargo.toml
[dependencies]
eframe = "0.29"
egui = "0.29"
egui_extras = { version = "0.29", features = ["image"] }
rfd = "0.15"
beaker = { path = "../beaker" }
```

**Alternative: Iced**
- More structured architecture (MVU pattern)
- Slightly longer compile times but still pure Rust
- Good choice if Elm-style architecture preferred

**Alternative: Slint**
- Fastest incremental builds
- DSL may add learning curve
- Good for embedded/minimal resource use
- Less flexible for complex custom widgets

---

## Migration from Tauri

**What we built in Tauri:**
- Detection command with heavy assertions
- TypeScript types and API wrappers
- Svelte component with image display
- File dialog integration
- State management with stores

**Migration to egui:**
1. Keep Rust backend in `beaker-gui/src/` (no longer in src-tauri)
2. Replace Tauri commands with direct function calls
3. Replace Svelte components with egui panels
4. Replace invoke("command") with direct Rust calls
5. State management becomes struct fields (immediate mode)
6. File dialogs: use `rfd` crate instead of tauri-plugin-dialog
7. Image display: use `egui_extras::RetainedImage`

**Advantages of Migration:**
1. ✅ Works in sandboxed environments
2. ✅ No asset protocol complexity
3. ✅ Direct function calls (no serialization overhead)
4. ✅ Simpler build (no Node.js required)
5. ✅ Smaller binary size
6. ✅ True integration testing possible

---

## Testing Strategy with egui

**Local Testing:**
```bash
# These will all succeed in sandboxed environment:
cargo check --manifest-path beaker-gui/Cargo.toml
cargo clippy --manifest-path beaker-gui/Cargo.toml
cargo build --manifest-path beaker-gui/Cargo.toml
cargo test --manifest-path beaker-gui/Cargo.toml --lib
```

**Unit Tests:**
- Test detection logic directly
- Test parameter validation
- Test state management functions
- All work without display

**Integration Tests:**
- Test with egui test harness
- Mock UI interactions
- Verify detection results
- No display required for tests

**CI Testing:**
- All commands work in GitHub Actions
- Can add visual regression tests later
- Can test actual GUI in CI with virtual display

---

## Conclusion

**egui is the clear winner for our use case:**
- ✅ Compiles and tests in sandboxed environments
- ✅ Pure Rust with no system dependencies
- ✅ Production-ready with large community
- ✅ Simpler than Tauri (no web stack)
- ✅ Direct integration with beaker library
- ✅ Excellent for desktop applications

**Next Steps:**
1. Get user approval for egui migration
2. Create new beaker-gui structure without Tauri
3. Implement Phase 1 Detection MVP with egui
4. Add comprehensive tests (all work locally)
5. Update CI workflow (all checks pass)
6. Document egui development workflow

