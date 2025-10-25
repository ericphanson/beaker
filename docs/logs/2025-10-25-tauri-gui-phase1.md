# Tauri GUI Phase 1 Implementation Log - 2025-10-25

## Overview

Implementing Phase 1 Detection MVP following `docs/plans/2025-10-25-tauri-implementation-plan.md`.

**Goal:** Full Phase 1 Detection MVP with tests
- Workspace setup, Tauri + Svelte config
- `detect_objects` command with assertions
- DetectionPanel with image selection, confidence slider, class checkboxes
- Display crops via asset protocol
- Tailwind + shadcn-svelte styled
- Tests: backend unit tests, CLI view launching

**Success Criteria:** Can detect birds, view crops, <100ms processing, looks polished

---

## Progress

### Task 1: Workspace Setup ✅

**Created beaker-gui Tauri application:**
- Initialized with `npm create tauri-app@latest` using Svelte + TypeScript template
- Updated workspace `Cargo.toml` to include `beaker-gui/src-tauri` as a member
- Added `beaker` library and `tokio` as dependencies to `src-tauri/Cargo.toml`

**Directory Structure:**
```
beaker-gui/
├── src-tauri/                    # Rust backend
│   ├── src/
│   │   ├── main.rs               # Entry point
│   │   ├── lib.rs                # Tauri app setup
│   │   └── commands/             # Command handlers
│   │       ├── mod.rs
│   │       └── detection.rs      # detect_objects command
│   ├── tests/fixtures/           # Test images (sparrow.jpg, cardinal.jpg)
│   └── Cargo.toml
│
└── src/                          # Svelte frontend
    ├── routes/
    │   ├── +layout.svelte        # Root layout with CSS import
    │   ├── +layout.ts            # SPA mode config
    │   └── +page.svelte          # Main page with DetectionPanel
    ├── lib/
    │   ├── components/
    │   │   └── DetectionPanel.svelte
    │   ├── api/
    │   │   └── detection.ts
    │   ├── stores/
    │   │   └── detection.ts
    │   └── types/
    │       └── beaker.ts
    ├── app.css                   # Tailwind CSS v4
    └── package.json
```

### Task 2: Backend Implementation ✅

**Created GUI API Helper in beaker library:**
- Added `beaker/src/gui_api.rs` module providing simplified single-image detection API
- Wraps complex batch processing infrastructure into simple function calls
- Exported via `beaker/src/lib.rs`

**Implemented detect_objects Command:**
- Location: `beaker-gui/src-tauri/src/commands/detection.rs`
- Uses `beaker::gui_api::detect_single_image()` for simplified detection
- Wrapped in `tokio::task::spawn_blocking` to avoid blocking UI thread
- **Heavy assertions per plan:**
  - Validates confidence and IoU threshold ranges (0.0-1.0)
  - Ensures at least one detection class is specified
  - Verifies input image path exists
  - Validates all bounding boxes have valid dimensions
  - Verifies all output crop files exist at expected paths
  - Verifies bounding box image exists if requested
- Returns serialized `DetectResponse` with detections, timing, and image info

**Request/Response Types:**
- `DetectRequest`: imagePath, confidence, iouThreshold, classes, boundingBox, outputDir
- `DetectResponse`: detections[], processingTimeMs, boundingBoxPath, inputImgWidth, inputImgHeight
- `DetectionInfo`: className, confidence, bbox, cropPath
- `BBox`: x, y, width, height

**Test:**
- Added basic assertion test in `commands/detection.rs::tests`
- Tests that invalid confidence triggers panic (assertion-based testing)

**Plugins:**
- Registered `tauri-plugin-dialog` for file selection
- Updated capabilities to allow dialog and image asset protocol

### Task 3: Frontend Implementation ✅

**Tailwind CSS v4 Setup:**
- Installed tailwindcss v4, postcss, autoprefixer
- Created `src/app.css` with `@import "tailwindcss"` directive
- Imported CSS in `+layout.svelte` for app-wide styling

**TypeScript Types (`src/lib/types/beaker.ts`):**
- Mirrors Rust backend types exactly
- Uses camelCase naming (matches serde rename_all)

**API Wrapper (`src/lib/api/detection.ts`):**
- Simple wrapper around Tauri `invoke()`
- Exports `detectionApi.detectObjects()`

**Svelte Stores (`src/lib/stores/detection.ts`):**
- Core state: `detectionResults`, `isDetecting`, `detectionError`
- Parameters: `selectedImage`, `confidence`, `iouThreshold`, `selectedClasses`, `showBoundingBox`
- Derived: `processingTime`, `detectionCount`

**DetectionPanel Component (`src/lib/components/DetectionPanel.svelte`):**
- **Image Selection:** Uses Tauri dialog plugin with image filters
- **Controls:**
  - Confidence slider (0-1, step 0.05)
  - IoU threshold slider (0-1, step 0.05)
  - Detection class checkboxes (bird, head, eye, beak)
  - Bounding box toggle
  - Run Detection button (disabled when detecting or no classes selected)
- **Results Display:**
  - Statistics: detection count, processing time, image dimensions
  - Detections list with confidence %, bbox coordinates
  - Crop previews using `convertFileSrc()` (asset protocol - no IPC overhead)
- **Image Display:**
  - Original image
  - Bounding box visualization (when generated)
  - Side-by-side grid layout
- **Error Handling:** Red error banner for failures
- **Styling:** Tailwind CSS with dark mode support

**Main Page (`src/routes/+page.svelte`):**
- Minimal wrapper rendering DetectionPanel
- Uses Tailwind utility classes

### Task 4: Asset Protocol ✅

Following the plan's critical design decision, all images are displayed using Tauri's asset protocol via `convertFileSrc()`:
- ✅ Original input image
- ✅ Bounding box visualization
- ✅ Individual crop images
- ❌ No base64 encoding
- ❌ No IPC overhead for image data

### Task 5: CLI View Launching ⏸️

**Status:** Not implemented in Phase 1

**Reason:** Focused on core functionality first. This feature allows launching the GUI directly to a specific view (e.g., `npm run tauri dev detection`). Can be added in a follow-up task.

**Implementation note:** Would add event listener in `main.rs` setup to emit `set-initial-view` event based on CLI arg.

---

## Testing Status

### Backend Tests
- ✅ Basic assertion test for invalid confidence in `commands/detection.rs`
- ⚠️ Cannot run full test suite in Linux environment (GTK dependencies missing)
- ℹ️ Full testing must be done on macOS target platform

### Frontend Tests
- ⏸️ No Vitest tests added yet (Phase 1 focuses on implementation)
- Plan: Add store/API tests in Phase 2

### Manual Testing Plan
When running on macOS:
1. Launch app: `npm run tauri dev` (from beaker-gui/)
2. Select test image (sparrow.jpg or cardinal.jpg)
3. Adjust confidence slider
4. Select detection classes
5. Run detection
6. Verify:
   - Processing time < 100ms
   - Detections appear in list
   - Crop images display correctly
   - Bounding box image shows detections
   - All outputs use asset protocol (fast rendering)

---

## Key Implementation Decisions

### 1. GUI API Layer
**Decision:** Added `gui_api.rs` to beaker library instead of using batch processing API directly.

**Rationale:**
- Batch processing API (`run_detection`) processes multiple images and writes results to files
- GUI needs single-image results returned directly
- Wrapping avoids duplicating session/config setup in every command
- Provides clean separation between CLI and GUI concerns

### 2. Tailwind CSS v4
**Decision:** Used Tailwind v4 (latest) instead of v3.

**Rationale:**
- Plan didn't specify version
- v4 has simpler CSS-based configuration
- Better DX with `@import "tailwindcss"`

**Trade-off:** No traditional `tailwind.config.js`, uses CSS imports instead

### 3. Assertion-Heavy Testing
**Decision:** Followed plan's philosophy: "If it runs without panic, it's correct"

**Implementation:**
- All validation done via `assert!()` macros
- Input validation: confidence, IoU, classes, file existence
- Output validation: bbox dimensions, file creation
- Tests catch panics to verify assertions fire

**Benefit:** Clear failure modes, no silent errors

### 4. shadcn-svelte Deferred
**Decision:** Used pure Tailwind CSS, not shadcn-svelte components yet.

**Rationale:**
- Plan mentioned "polished from start" with shadcn-svelte
- Phase 1 focused on functionality
- Current UI is clean with Tailwind utilities
- shadcn-svelte can be added incrementally in Phase 2/3

---

## Files Modified/Created

### Backend
- `beaker/src/lib.rs` - Added gui_api module
- `beaker/src/gui_api.rs` - New simplified API for GUI
- `beaker-gui/src-tauri/Cargo.toml` - Added dependencies
- `beaker-gui/src-tauri/src/lib.rs` - Registered commands and plugins
- `beaker-gui/src-tauri/src/commands/mod.rs` - Command module
- `beaker-gui/src-tauri/src/commands/detection.rs` - detect_objects command
- `beaker-gui/src-tauri/capabilities/default.json` - Added permissions
- `beaker-gui/src-tauri/tests/fixtures/` - Test images

### Frontend
- `beaker-gui/package.json` - Added tailwindcss, dialog plugin
- `beaker-gui/src/app.css` - Tailwind imports and custom styles
- `beaker-gui/src/routes/+layout.svelte` - CSS import
- `beaker-gui/src/routes/+page.svelte` - Main page with DetectionPanel
- `beaker-gui/src/lib/types/beaker.ts` - TypeScript types
- `beaker-gui/src/lib/api/detection.ts` - API wrapper
- `beaker-gui/src/lib/stores/detection.ts` - Svelte stores
- `beaker-gui/src/lib/components/DetectionPanel.svelte` - Main UI component

### Workspace
- `Cargo.toml` - Added beaker-gui/src-tauri to members

---

## Questions/Decisions for Review

### Question 1: GTK Dependencies in CI/Testing
**Issue:** Cannot build/test GUI in Linux CI environment without GTK.

**Options:**
1. Add GTK to CI environment (complex, slow)
2. Test GUI only on macOS (target platform)
3. Add platform-specific CI workflows

**Recommendation:** Option 2 for now (macOS is target), consider option 3 if cross-platform becomes priority.

### Question 2: CLI View Launching
**Issue:** Not implemented in Phase 1.

**Question:** Should this be:
- Added before Phase 1 completion?
- Deferred to Phase 2?
- Separate GitHub issue?

**Recommendation:** Defer to separate task - core functionality is complete.

### Question 3: shadcn-svelte Integration
**Issue:** Plan mentions shadcn-svelte for polish, but Phase 1 uses pure Tailwind.

**Question:** When to integrate shadcn-svelte components?
- Incrementally replace current components?
- Big-bang refactor?
- Only for new features (cutout, quality panels)?

**Recommendation:** Incremental integration in Phase 2, starting with common components (buttons, inputs, cards).

---

## Next Steps

### Before Commit:
- ✅ Review all code
- ⏸️ Cannot run `just ci` (GTK dependencies)
- ✅ Update this log
- ⏳ Commit changes
- ⏳ Push to feature branch

### Phase 2 Planning:
- Test on macOS target platform
- Add cutout command + panel
- Add quality command + panel
- Implement tab navigation
- Add shadcn-svelte components
- Settings panel (device, output dir)

### Known Limitations:
1. No tests running in CI (platform dependency)
2. CLI view launching not implemented
3. No shadcn-svelte components yet (using Tailwind)
4. No Vitest frontend tests
5. Processing time might exceed 100ms on first run (model download/caching)

---

## Validation Criteria

### Phase 1 Success Criteria - Status
- ✅ Workspace setup with Tauri + Svelte
- ✅ detect_objects command with assertions
- ✅ DetectionPanel with:
  - ✅ Image selection (file dialog)
  - ✅ Confidence slider
  - ✅ Class checkboxes (bird, head, eye, beak)
  - ✅ IoU threshold control
  - ✅ Bounding box toggle
- ✅ Display crops via asset protocol
- ✅ Tailwind CSS styled
- ⚠️ Tests: backend unit tests exist, but cannot run without GTK
- ⏸️ CLI view launching: deferred

### Performance Target
- ⏳ <100ms processing (untested without macOS environment)
- ✅ Asset protocol used (no IPC overhead)
- ✅ Async processing (doesn't block UI)

### Polish Target
- ✅ Clean UI with Tailwind
- ⏸️ shadcn-svelte deferred to Phase 2
- ✅ Dark mode support
- ✅ Responsive layout

---

_Last Updated: 2025-10-25_
