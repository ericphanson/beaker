# Beaker Tauri GUI Implementation Plan

**Date:** 2025-10-25
**Framework:** Tauri 2.0 + Svelte 5 + TypeScript
**Timeline:** 6-8 weeks (2 weeks MVP, 4-6 weeks full features)

---

## 1. Architecture

### Workspace Structure

```
beaker/
├── Cargo.toml                    # Workspace: ["beaker", "beaker-gui/src-tauri"]
├── beaker/                       # Core lib (existing)
│   └── src/lib.rs                # Public API for GUI
│
└── beaker-gui/                   # NEW: Tauri app
    ├── src-tauri/                # Rust backend
    │   ├── src/
    │   │   ├── main.rs
    │   │   └── commands/         # Tauri command handlers
    │   │       ├── detection.rs
    │   │       ├── cutout.rs
    │   │       └── quality.rs
    │   └── tests/
    │       └── fixtures/         # sparrow.jpg, cardinal.jpg, etc.
    │
    └── src/                      # Svelte frontend
        ├── App.svelte
        └── lib/
            ├── components/       # DetectionPanel.svelte, etc.
            ├── stores/           # Reactive state (detection.ts, etc.)
            ├── api/              # Tauri invoke wrappers
            └── types/            # TypeScript types
```

### Tech Stack

- **Backend:** Tauri 2.0, async commands (tokio), beaker lib
- **Frontend:** Svelte 5 (simpler than React), TypeScript, Vite
- **Styling:** Tailwind CSS + shadcn-svelte (polished from start)
- **Images:** Asset protocol (`convertFileSrc`) - no IPC overhead

---

## 2. Cookie-Cutter Pattern: Adding Features

When beaker lib gets a new algorithm (e.g., "species classification"), follow this pattern:

### 2.1 Backend (15 min)
```rust
// src-tauri/src/commands/species.rs
#[derive(Deserialize)]
pub struct ClassifyRequest {
    pub image_path: String,
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    pub predictions: Vec<SpeciesPrediction>,
    pub processing_time_ms: f64,
}

#[tauri::command]
pub async fn classify_species(request: ClassifyRequest) -> Result<ClassifyResponse, String> {
    // Assertions
    assert!(request.top_k > 0 && request.top_k <= 10);
    assert!(Path::new(&request.image_path).exists());

    // Call beaker lib
    tokio::task::spawn_blocking(move || {
        let classifier = beaker::species::SpeciesClassifier::new()?;
        let result = classifier.classify(&request.image_path, request.top_k)?;

        Ok(ClassifyResponse {
            predictions: result.predictions,
            processing_time_ms: result.processing_time_ms,
        })
    }).await.map_err(|e| e.to_string())?
}
```

### 2.2 Frontend (45 min)

**Types** (5 min):
```typescript
// src/lib/types/beaker.ts
export interface ClassifyRequest { imagePath: string; topK: number; }
export interface ClassifyResponse { predictions: SpeciesPrediction[]; }
```

**API** (5 min):
```typescript
// src/lib/api/species.ts
export const speciesApi = {
  async classify(req: ClassifyRequest): Promise<ClassifyResponse> {
    return await invoke('classify_species', { request: req });
  },
};
```

**Store** (10 min):
```typescript
// src/lib/stores/species.ts
export const speciesResults = writable<ClassifyResponse | null>(null);
export const isClassifying = writable(false);
```

**Component** (25 min):
```svelte
<!-- src/lib/components/SpeciesPanel.svelte -->
<script lang="ts">
  import { speciesApi } from '$lib/api/species';
  // ... controls, image viewer, results display
</script>
```

**Wire up** (2 min): Add tab to `App.svelte`

**Total: ~1 hour per feature**

---

## 3. Testing Strategy

### 3.1 Philosophy: Assert-Heavy + CLI Testing

**No browser automation needed.** Test via:
1. **Rust unit tests** (70%) - Commands are functions, test directly
2. **Frontend tests** (20%) - Vitest for stores/API/utils
3. **Assertions everywhere** - If it runs without panic, it's correct
4. **CLI view launching** - Fast manual testing

### 3.2 Backend Tests

```rust
// src-tauri/src/commands/detection.rs
#[tauri::command]
pub async fn detect_objects(request: DetectRequest) -> Result<DetectResponse, String> {
    // Heavy assertions
    assert!(request.confidence >= 0.0 && request.confidence <= 1.0);
    assert!(!request.classes.is_empty());
    assert!(Path::new(&request.image_path).exists());

    let result = pipeline.process_image(&request.image_path)?;

    // Validate outputs
    for det in &result.detections {
        assert!(det.bbox.width > 0.0 && det.bbox.height > 0.0);
    }

    // Verify files created
    for path in &output_paths {
        assert!(Path::new(path).exists());
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_detect_objects() {
        let req = DetectRequest {
            image_path: "tests/fixtures/sparrow.jpg".to_string(),
            confidence: 0.5,
            classes: vec!["bird".to_string()],
        };

        let result = detect_objects(req).await;
        assert!(result.is_ok());
        assert!(!result.unwrap().detections.is_empty());
    }
}
```

**Run:** `cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml`

### 3.3 Frontend Tests

```typescript
// src/lib/stores/detection.test.ts
import { get } from 'svelte/store';
import { detectionParams, processingTime } from './detection';

it('should compute processing time', () => {
  detectionResults.set({ processingTimeMs: 123.45, /* ... */ });
  expect(get(processingTime)).toBe('123ms');
});
```

**Run:** `npm test` (Vitest)

### 3.4 CLI View Testing

**Launch to specific views:**
```bash
npm run tauri dev detection  # Jump to detection tab
npm run tauri dev cutout     # Jump to cutout tab
./target/release/beaker-gui quality  # Production build
```

**Implementation:**
```rust
// main.rs
.setup(|app| {
    let view = std::env::args().nth(1);
    if let Some(v) = view {
        app.get_webview_window("main").unwrap().emit("set-initial-view", v).unwrap();
    }
    Ok(())
})
```

**Automated testing:**
```bash
#!/bin/bash
# scripts/test-views.sh
for view in detection cutout quality batch; do
    timeout 10 npm run tauri dev -- "$view" || {
        [ $? -eq 124 ] && echo "✅ $view OK" || exit 1
    }
done
```

### 3.5 Coverage Goals

- Backend commands: **90%+**
- Frontend stores/API: **80%+**
- Manual smoke tests: **15 min checklist** before releases

---

## 4. MVP Phases

### Phase 1: Detection MVP (Week 1-2)
- [ ] Workspace setup, Tauri + Svelte config
- [ ] `detect_objects` command with assertions
- [ ] DetectionPanel with image selection, confidence slider, class checkboxes
- [ ] Display crops via asset protocol
- [ ] Tailwind + shadcn-svelte styled
- [ ] Tests: backend unit tests, CLI view launching

**Success:** Can detect birds, view crops, <100ms processing, looks polished

### Phase 2: Core Features (Week 3-4)
- [ ] Cutout command + panel (alpha matting controls)
- [ ] Quality command + panel (heatmap visualization)
- [ ] Settings panel (device, output dir)
- [ ] Tab navigation between tools

### Phase 3: Batch & Polish (Week 5-8)
- [ ] Batch queue, progress tracking, parallel processing
- [ ] Drag-and-drop, keyboard shortcuts, dark mode
- [ ] Recent files, export presets, metadata viewer

---

## 5. Key Design Decisions

### Asset Protocol (Critical)
```typescript
// ✅ Fast: Use asset protocol for images
import { convertFileSrc } from '@tauri-apps/api/core';
const url = convertFileSrc('/path/to/crop.jpg');
<img src={url} />

// ❌ Slow: Don't send images over IPC
const base64 = await invoke('get_image_base64');  // Bad!
```

### Async Processing
```rust
#[tauri::command]
async fn detect(req: DetectRequest) -> Result<DetectResponse, String> {
    tokio::task::spawn_blocking(move || {
        // ONNX inference doesn't block UI thread
        pipeline.process_image(&req.image_path)
    }).await.map_err(|e| e.to_string())?
}
```

### State Management
- **Svelte stores** for reactive state
- **Assertions** in store subscribers
- **No Redux/complex state** - Svelte stores are simple

---

## 6. Quick Reference

### Commands
```bash
# Dev
npm run tauri dev [view]       # Optional: detection, cutout, quality

# Test
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml
npm test
./scripts/test-views.sh

# Build
npm run tauri build
```

### File Organization
| Path | Purpose |
|------|---------|
| `src-tauri/src/commands/` | Tauri command handlers (backend) |
| `src/lib/api/` | Invoke wrappers (frontend calls backend) |
| `src/lib/stores/` | Reactive state (Svelte stores) |
| `src/lib/components/` | UI components (panels, viewers) |
| `src-tauri/tests/` | Backend unit tests + fixtures |

### Adding a Feature Checklist
- [ ] Backend command in `src-tauri/src/commands/feature.rs`
- [ ] Types in `src/lib/types/beaker.ts`
- [ ] API wrapper in `src/lib/api/feature.ts`
- [ ] Store in `src/lib/stores/feature.ts`
- [ ] Component in `src/lib/components/FeaturePanel.svelte`
- [ ] Wire up in `App.svelte`
- [ ] Tests in `src-tauri/src/commands/feature.rs` (unit)
- [ ] CLI view test: `npm run tauri dev feature`

---

**Philosophy:** Performant from start (async + asset protocol), styled from start (Tailwind + shadcn), tested from start (assertions + CLI), easy to extend (cookie-cutter pattern).

**Document Version:** 2.0 (Condensed)
**Last Updated:** 2025-10-25
