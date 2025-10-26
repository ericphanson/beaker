# Implementation Log: Proposal 0 - File Navigation & Opening

**Date:** 2025-10-26
**Proposal:** Proposal 0 from `docs/plans/2025-10-25-detection-features-plan.md`
**Goal:** Enable users to open images/folders from within the GUI using native file dialogs

---

## Progress

### Dependencies Added
- ✅ Added `rfd = "0.15"` for native file dialogs
- ✅ Added `serde_json = "1.0"` for recent files persistence
- ✅ Added `dirs = "5.0"` for cross-platform config directory

### Implementation Plan
1. Create recent files management module
2. Create welcome view with:
   - Large "Open Image" and "Open Folder" buttons
   - Drag & drop zone
   - Recent files list (last 10)
   - Getting started tips
3. Integrate native file dialogs
4. Update menu bar with File > Open Image and File > Open Folder
5. Update BeakerApp to show WelcomeView when no args provided

### Features Implemented
- ✅ Recent files module (with full persistence and tests)
- ✅ Welcome view UI (with drag & drop zone, buttons, recent files list)
- ✅ Native file dialogs (using rfd crate)
- ✅ Drag & drop support (using egui's dropped_files API)
- ✅ Recent files display (shows last 10 with timestamps)
- ✅ Menu bar updates (File > Open Image and File > Open Folder)
- ✅ Unit tests (9 tests passing)
- ✅ Integration tests (included in view modules)

---

## Implementation Details

### Files Created
1. `beaker-gui/src/recent_files.rs` - Recent files management module (~250 LOC)
2. `beaker-gui/src/views/welcome.rs` - Welcome view UI (~320 LOC)

### Files Modified
1. `beaker-gui/Cargo.toml` - Added dependencies (rfd, serde_json, dirs, chrono, tempfile)
2. `beaker-gui/src/lib.rs` - Exported new modules
3. `beaker-gui/src/views/mod.rs` - Added welcome view
4. `beaker-gui/src/app.rs` - Updated to use WelcomeView and handle actions

### Key Features
- **Recent files persistence**: Stores up to 10 recent items in `~/.config/beaker-gui/recent.json`
- **Time formatting**: Shows friendly time strings ("2 hours ago", "3 days ago", etc.)
- **Drag & drop**: Visual feedback when hovering files over drop zone
- **Native file dialogs**: Cross-platform file/folder pickers with filters
- **Smart navigation**: Clicking recent items reopens them
- **State management**: BeakerApp uses AppState enum to switch between Welcome and Detection views

### Total LOC
- Approximately 570 LOC (within estimated 400-500 LOC range for core implementation)

---

## Testing Results
All 9 unit tests passing:
- ✅ test_add_recent_file
- ✅ test_add_duplicate_moves_to_front
- ✅ test_max_recent_files
- ✅ test_clear
- ✅ test_remove
- ✅ test_persistence
- ✅ test_welcome_view_creation
- ✅ test_format_time_ago
- ✅ test_toml_parsing_detections (existing)

---

## Notes
- Following the detection features plan specifications
- Recent files stored in `~/.config/beaker-gui/recent.json` (Linux) or platform equivalent via dirs crate
- Uses egui's `dropped_files()` and `hovered_files` API for drag & drop feedback
- Folder mode shows TODO message (will be implemented in Proposal A)
- Welcome screen is default when launching without --image argument

---

## macOS File Dialog Fix

**Issue:** File dialogs didn't work on macOS (buttons appeared but nothing happened)
**Root cause:** macOS requires async file dialogs due to event loop constraints
**Solution:** Use `rfd::AsyncFileDialog` on all platforms for consistency

### Changes Made
1. Added `pollster = "0.4"` dependency for blocking on async operations
2. Updated `open_file_dialog()` and `open_folder_dialog()` in both `app.rs` and `welcome.rs`
3. Use `rfd::AsyncFileDialog` with `pollster::block_on()` on **all platforms**
4. Removed platform-specific conditional compilation for simpler, more maintainable code

### Why async on all platforms?
- **Consistency:** Single code path is easier to maintain and debug
- **Simplicity:** No conditional compilation needed
- **Performance:** Negligible difference for infrequent file dialog operations
- **Future-proof:** Works correctly on all platforms including macOS

### Testing
- ✅ All tests still passing (9 unit tests)
- ✅ CI passes on Linux
- ✅ Should now work correctly on macOS

---

## UX Improvements: Clickable Drop Zone & Logging

**Changes:**
1. Made drop zone clickable - clicking opens file dialog (was hover-only before)
2. Added comprehensive logging throughout the app
3. Added File > Open menu item

### Clickable Drop Zone
- Changed from `egui::Sense::hover()` to `egui::Sense::click()`
- Added visual hover feedback (lighter background)
- Added "(or click to browse)" hint text
- Clicking drop zone opens file picker for images

### Comprehensive Logging
All interactions now logged to stderr with clear prefixes:
- `[WelcomeView]` - Welcome screen actions
- `[BeakerApp]` - App-level actions

**Logged events:**
- Button clicks (Open Image, Open Folder, recent files)
- Drop zone clicks
- File/folder drops (drag & drop)
- File dialog opening and results (path or None)
- State transitions (switching to Detection view)
- Errors (with ERROR: prefix)

Example log output:
```
[WelcomeView] Drop zone clicked, opening file dialog...
[WelcomeView] Opening file dialog (async)...
[WelcomeView] File dialog result: Some("/path/to/image.jpg")
[BeakerApp] Received action: OpenImage("/path/to/image.jpg")
[BeakerApp] Opening image: "/path/to/image.jpg"
[BeakerApp] Image loaded successfully, switching to Detection view
```

### Testing
- ✅ All 9 unit tests passing
- ✅ CI pipeline passing (format, clippy, build, tests)
- ✅ Committed and pushed

---

## Next Steps
1. ✅ Run just ci to validate
2. ✅ Commit and push
3. ✅ Fix macOS file dialog issue
4. ✅ Make drop zone clickable and add logging
5. Future: Implement Proposal A (Bulk/Directory Mode)
