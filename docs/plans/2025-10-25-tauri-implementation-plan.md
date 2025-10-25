# Beaker Tauri GUI Implementation Plan

**Date:** 2025-10-25
**Status:** Implementation Specification
**Framework:** Tauri 2.0 + Svelte 5 + TypeScript

---

## Executive Summary

This document provides a complete implementation plan for building Beaker's GUI using Tauri. The architecture focuses on:

1. **Multi-crate workspace** - Separate `beaker-gui` Tauri app depending on `beaker` lib
2. **Performant & styled from start** - shadcn-svelte components, Tailwind CSS
3. **Phased MVP delivery** - Detection → Cutout → Quality → Advanced features
4. **Cookie-cutter feature addition** - Clear pattern for adding new algorithms

**Timeline:** 6-8 weeks total (2 weeks MVP + 4-6 weeks full features)

---

## 1. Workspace Structure

### 1.1 Directory Layout

```
beaker/
├── Cargo.toml                           # Workspace root
│   [workspace]
│   members = ["beaker", "beaker-gui/src-tauri"]
│   resolver = "2"
│
├── beaker/                              # Core library (existing)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                       # Public API
│   │   ├── core/                        # Detection, cutout, quality
│   │   ├── shared/                      # Metadata, I/O
│   │   └── ...
│   └── tests/
│
├── beaker-gui/                          # NEW: Tauri desktop app
│   ├── src-tauri/                       # Rust backend
│   │   ├── Cargo.toml
│   │   ├── tauri.conf.json              # Tauri configuration
│   │   ├── build.rs
│   │   ├── src/
│   │   │   ├── main.rs                  # Tauri app entry
│   │   │   ├── commands/                # Tauri command handlers
│   │   │   │   ├── mod.rs
│   │   │   │   ├── detection.rs
│   │   │   │   ├── cutout.rs
│   │   │   │   ├── quality.rs
│   │   │   │   └── filesystem.rs
│   │   │   ├── state/                   # Application state
│   │   │   │   ├── mod.rs
│   │   │   │   └── app_state.rs
│   │   │   └── error.rs                 # Error handling
│   │   ├── icons/
│   │   └── capabilities/
│   │
│   ├── src/                             # Frontend (Svelte)
│   │   ├── main.ts                      # Entry point
│   │   ├── App.svelte                   # Root component
│   │   ├── lib/
│   │   │   ├── components/              # UI components
│   │   │   │   ├── ui/                  # shadcn-svelte components
│   │   │   │   │   ├── button.svelte
│   │   │   │   │   ├── slider.svelte
│   │   │   │   │   ├── card.svelte
│   │   │   │   │   └── ...
│   │   │   │   ├── DetectionPanel.svelte
│   │   │   │   ├── CutoutPanel.svelte
│   │   │   │   ├── QualityPanel.svelte
│   │   │   │   ├── ImageViewer.svelte
│   │   │   │   ├── ResultsGrid.svelte
│   │   │   │   └── SettingsPanel.svelte
│   │   │   ├── stores/                  # Svelte stores (state)
│   │   │   │   ├── detection.ts
│   │   │   │   ├── cutout.ts
│   │   │   │   ├── quality.ts
│   │   │   │   └── settings.ts
│   │   │   ├── api/                     # Tauri API wrappers
│   │   │   │   ├── detection.ts
│   │   │   │   ├── cutout.ts
│   │   │   │   ├── quality.ts
│   │   │   │   └── filesystem.ts
│   │   │   ├── types/                   # TypeScript types
│   │   │   │   └── beaker.ts
│   │   │   └── utils/
│   │   │       ├── convertFileSrc.ts    # Asset protocol helper
│   │   │       └── formatters.ts
│   │   └── assets/
│   │
│   ├── package.json                     # Frontend dependencies
│   ├── vite.config.ts                   # Vite configuration
│   ├── tsconfig.json                    # TypeScript config
│   ├── tailwind.config.js               # Tailwind CSS config
│   └── svelte.config.js                 # Svelte config
│
├── docs/
│   └── plans/
│       ├── 2025-10-25-gui-plan.md
│       └── 2025-10-25-tauri-implementation-plan.md  # This document
│
└── README.md
```

---

## 2. Technology Stack

### 2.1 Backend (Tauri/Rust)

```toml
# beaker-gui/src-tauri/Cargo.toml
[package]
name = "beaker-gui"
version = "0.1.0"
edition = "2021"

[dependencies]
beaker = { path = "../../beaker" }
tauri = { version = "2.1", features = ["protocol-asset"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tokio = { version = "1", features = ["rt-multi-thread"] }

[build-dependencies]
tauri-build = { version = "2.0" }
```

### 2.2 Frontend (Svelte 5 + TypeScript)

```json
{
  "name": "beaker-gui",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "tauri": "tauri"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.1.0",
    "@tauri-apps/plugin-dialog": "^2.0.0",
    "@tauri-apps/plugin-fs": "^2.0.0",
    "svelte": "^5.0.0"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@tauri-apps/cli": "^2.1.0",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "^5.6.0",
    "vite": "^6.0.0"
  }
}
```

**Why Svelte 5?**
- Simpler than React (less boilerplate)
- Excellent performance (compiles to vanilla JS)
- Runes API for reactive state (similar to React hooks)
- Smaller bundle sizes
- Great TypeScript support

**Alternative: React** (if team prefers)
- More familiar to many developers
- Larger ecosystem of components
- Slightly more verbose

---

## 3. Tauri Configuration

### 3.1 tauri.conf.json

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Beaker",
  "version": "0.1.0",
  "identifier": "com.beaker.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:5173"
  },
  "app": {
    "windows": [
      {
        "title": "Beaker - Bird Image Analysis",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": "default-src 'self' ipc: http://ipc.localhost; img-src 'self' asset: http://asset.localhost data: blob:; style-src 'self' 'unsafe-inline'",
      "assetProtocol": {
        "enable": true,
        "scope": ["**"]
      }
    }
  },
  "bundle": {
    "active": true,
    "targets": ["dmg", "deb", "msi"],
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  }
}
```

**Key Configuration Points:**

1. **Asset Protocol** - Enabled with `"scope": ["**"]` to serve processed images
2. **CSP** - Allows asset protocol for image loading
3. **Window size** - 1200x800 default, 800x600 minimum
4. **Bundle targets** - DMG (macOS), DEB (Linux), MSI (Windows)

---

## 4. Styling Strategy (From the Start)

### 4.1 Tailwind CSS Setup

```js
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))'
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)'
      }
    }
  },
  plugins: []
}
```

### 4.2 Theme Variables (CSS)

```css
/* src/app.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --primary: 221.2 83.2% 53.3%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --border: 214.3 31.8% 91.4%;
    --radius: 0.5rem;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --primary: 217.2 91.2% 59.8%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
  }
}
```

### 4.3 Component Library (shadcn-svelte approach)

**Install key components:**
```bash
npx shadcn-svelte@latest init
npx shadcn-svelte@latest add button slider card input label tabs
```

**Result:** Pre-styled, accessible components in `src/lib/components/ui/`

---

## 5. Core Patterns: Cookie-Cutter Feature Addition

### 5.1 Pattern Overview

When beaker lib gets a new feature (e.g., "species classification"), follow this pattern:

```
1. Rust backend (Tauri command) - src-tauri/src/commands/species.rs
2. TypeScript types - src/lib/types/beaker.ts
3. API wrapper - src/lib/api/species.ts
4. Svelte store - src/lib/stores/species.ts
5. UI component - src/lib/components/SpeciesPanel.svelte
6. Wire up in App.svelte
```

**Time per feature:** ~2-4 hours once pattern is established

---

### 5.2 Pattern: Tauri Command (Backend)

**File:** `src-tauri/src/commands/detection.rs`

```rust
use beaker::core::detection::{DetectionPipeline, DetectionConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::State;

// ============================================================================
// STEP 1: Define request/response types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DetectRequest {
    pub image_path: String,
    pub confidence: f32,
    pub classes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DetectResponse {
    pub detections: Vec<Detection>,
    pub output_paths: Vec<String>, // Paths to saved crops
    pub processing_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Detection {
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

#[derive(Debug, Serialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

// ============================================================================
// STEP 2: Implement Tauri command
// ============================================================================

#[tauri::command]
pub async fn detect_objects(
    request: DetectRequest,
) -> Result<DetectResponse, String> {
    // Create config from request
    let config = DetectionConfig {
        confidence: request.confidence,
        classes: request.classes,
        ..Default::default()
    };

    // Call beaker library
    let pipeline = DetectionPipeline::new(config)
        .map_err(|e| e.to_string())?;

    let result = pipeline.process_image(&request.image_path)
        .map_err(|e| e.to_string())?;

    // Save crops to output directory
    let output_paths = save_crops(&result)?;

    // Convert to response format
    Ok(DetectResponse {
        detections: result.detections.into_iter().map(|d| Detection {
            class_name: d.class_name,
            confidence: d.confidence,
            bbox: BoundingBox {
                x: d.bbox.x,
                y: d.bbox.y,
                width: d.bbox.width,
                height: d.bbox.height,
            },
        }).collect(),
        output_paths,
        processing_time_ms: result.processing_time_ms,
    })
}

fn save_crops(result: &beaker::DetectionResult) -> Result<Vec<String>, String> {
    // Save crops to temporary directory or user-specified output
    // Return paths as strings for frontend
    todo!("Implement crop saving")
}
```

**Register command in main.rs:**

```rust
// src-tauri/src/main.rs
mod commands;

use commands::detection;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            detection::detect_objects,
            // Add more commands here
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

### 5.3 Pattern: TypeScript Types

**File:** `src/lib/types/beaker.ts`

```typescript
// ============================================================================
// STEP 3: Define TypeScript types matching Rust types
// ============================================================================

export interface DetectRequest {
  imagePath: string;
  confidence: number;
  classes: string[];
}

export interface DetectResponse {
  detections: Detection[];
  outputPaths: string[]; // Paths to crops (will use convertFileSrc)
  processingTimeMs: number;
}

export interface Detection {
  className: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Enum for detection classes
export const DetectionClass = {
  BIRD: 'bird',
  HEAD: 'head',
  EYE: 'eye',
  BEAK: 'beak',
} as const;

export type DetectionClassType = typeof DetectionClass[keyof typeof DetectionClass];
```

---

### 5.4 Pattern: API Wrapper

**File:** `src/lib/api/detection.ts`

```typescript
import { invoke } from '@tauri-apps/api/core';
import type { DetectRequest, DetectResponse } from '$lib/types/beaker';

// ============================================================================
// STEP 4: Create API wrapper for Tauri commands
// ============================================================================

export const detectionApi = {
  /**
   * Detect objects in an image
   */
  async detect(request: DetectRequest): Promise<DetectResponse> {
    return await invoke<DetectResponse>('detect_objects', { request });
  },
};

// Export convenience functions
export async function detectBirds(
  imagePath: string,
  confidence: number = 0.5,
  classes: string[] = ['bird', 'head']
): Promise<DetectResponse> {
  return detectionApi.detect({
    imagePath,
    confidence,
    classes,
  });
}
```

---

### 5.5 Pattern: Svelte Store (State Management)

**File:** `src/lib/stores/detection.ts`

```typescript
import { writable, derived } from 'svelte/store';
import type { DetectResponse } from '$lib/types/beaker';

// ============================================================================
// STEP 5: Create Svelte stores for reactive state
// ============================================================================

// Detection parameters
export const detectionParams = writable({
  confidence: 0.5,
  classes: ['bird', 'head'],
});

// Current image being processed
export const currentImage = writable<string | null>(null);

// Detection results
export const detectionResults = writable<DetectResponse | null>(null);

// Loading state
export const isDetecting = writable(false);

// Error state
export const detectionError = writable<string | null>(null);

// Derived store: formatted processing time
export const processingTime = derived(
  detectionResults,
  ($results) => {
    if (!$results) return null;
    return `${$results.processingTimeMs.toFixed(0)}ms`;
  }
);

// Derived store: detection count
export const detectionCount = derived(
  detectionResults,
  ($results) => $results?.detections.length ?? 0
);
```

---

### 5.6 Pattern: UI Component

**File:** `src/lib/components/DetectionPanel.svelte`

```svelte
<script lang="ts">
  import { Button } from '$lib/components/ui/button';
  import { Slider } from '$lib/components/ui/slider';
  import { Label } from '$lib/components/ui/label';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { open } from '@tauri-apps/plugin-dialog';
  import { convertFileSrc } from '@tauri-apps/api/core';
  import { detectionApi } from '$lib/api/detection';
  import {
    detectionParams,
    currentImage,
    detectionResults,
    isDetecting,
    detectionError,
    processingTime,
  } from '$lib/stores/detection';

  // ========================================================================
  // STEP 6: Build UI component using stores and API
  // ========================================================================

  // Reactive state from stores
  let confidence = $state($detectionParams.confidence);
  let selectedClasses = $state($detectionParams.classes);

  // Handle file selection
  async function selectImage() {
    const selected = await open({
      multiple: false,
      filters: [
        { name: 'Images', extensions: ['jpg', 'jpeg', 'png'] },
      ],
    });

    if (selected) {
      currentImage.set(selected);
    }
  }

  // Run detection
  async function runDetection() {
    if (!$currentImage) return;

    isDetecting.set(true);
    detectionError.set(null);

    try {
      const result = await detectionApi.detect({
        imagePath: $currentImage,
        confidence,
        classes: selectedClasses,
      });

      detectionResults.set(result);
    } catch (err) {
      detectionError.set(err instanceof Error ? err.message : String(err));
    } finally {
      isDetecting.set(false);
    }
  }

  // Update store when params change
  $effect(() => {
    detectionParams.set({ confidence, classes: selectedClasses });
  });
</script>

<!-- UI Template -->
<div class="grid grid-cols-2 gap-4 p-4">
  <!-- Left: Image & Results -->
  <div class="space-y-4">
    <Card>
      <CardHeader>
        <CardTitle>Image</CardTitle>
      </CardHeader>
      <CardContent>
        {#if $currentImage}
          <img
            src={convertFileSrc($currentImage)}
            alt="Selected"
            class="w-full h-auto rounded-lg"
          />
        {:else}
          <div class="h-64 bg-muted rounded-lg flex items-center justify-center">
            <p class="text-muted-foreground">No image selected</p>
          </div>
        {/if}
      </CardContent>
    </Card>

    {#if $detectionResults}
      <Card>
        <CardHeader>
          <CardTitle>Results ({$detectionResults.detections.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div class="grid grid-cols-3 gap-2">
            {#each $detectionResults.outputPaths as path}
              <img
                src={convertFileSrc(path)}
                alt="Crop"
                class="w-full h-auto rounded border"
              />
            {/each}
          </div>
          {#if $processingTime}
            <p class="text-sm text-muted-foreground mt-2">
              Processing time: {$processingTime}
            </p>
          {/if}
        </CardContent>
      </Card>
    {/if}
  </div>

  <!-- Right: Controls -->
  <Card>
    <CardHeader>
      <CardTitle>Detection Settings</CardTitle>
    </CardHeader>
    <CardContent class="space-y-4">
      <div class="space-y-2">
        <Button onclick={selectImage} class="w-full">
          Select Image
        </Button>
      </div>

      <div class="space-y-2">
        <Label for="confidence">Confidence: {confidence.toFixed(2)}</Label>
        <Slider
          id="confidence"
          min={0}
          max={1}
          step={0.05}
          bind:value={confidence}
        />
      </div>

      <div class="space-y-2">
        <Label>Detect Classes</Label>
        <div class="flex flex-col gap-2">
          {#each ['bird', 'head', 'eye', 'beak'] as cls}
            <label class="flex items-center gap-2">
              <input
                type="checkbox"
                bind:group={selectedClasses}
                value={cls}
                class="rounded"
              />
              <span class="capitalize">{cls}</span>
            </label>
          {/each}
        </div>
      </div>

      <Button
        onclick={runDetection}
        disabled={!$currentImage || $isDetecting}
        class="w-full"
      >
        {$isDetecting ? 'Detecting...' : 'Detect'}
      </Button>

      {#if $detectionError}
        <p class="text-sm text-destructive">{$detectionError}</p>
      {/if}
    </CardContent>
  </Card>
</div>
```

---

## 6. MVP Phasing

### Phase 1: Detection MVP (Week 1-2)

**Goal:** Minimal viable detection interface

**Features:**
- ✅ Select single image (file dialog)
- ✅ Adjust confidence slider
- ✅ Select classes to detect
- ✅ Run detection
- ✅ Display crops in grid
- ✅ Show processing time

**Deliverables:**
- Tauri commands: `detect_objects`
- UI components: `DetectionPanel.svelte`
- Basic styling with Tailwind + shadcn-svelte
- Asset protocol working for image display

**Success Criteria:**
- Can detect birds in <100ms per image
- UI feels responsive (no blocking)
- Crops display correctly via asset protocol
- Looks polished (not prototype-y)

---

### Phase 2: Full Core Features (Week 3-4)

**Goal:** All three core tools (Detection, Cutout, Quality)

**New Features:**
- ✅ Background removal (cutout) panel
- ✅ Quality assessment panel
- ✅ Tab navigation between tools
- ✅ Settings panel (output dir, device selection)

**New Tauri Commands:**
```rust
// src-tauri/src/commands/cutout.rs
#[tauri::command]
async fn remove_background(request: CutoutRequest) -> Result<CutoutResponse, String>;

// src-tauri/src/commands/quality.rs
#[tauri::command]
async fn assess_quality(request: QualityRequest) -> Result<QualityResponse, String>;

// src-tauri/src/commands/filesystem.rs
#[tauri::command]
async fn select_output_directory() -> Result<String, String>;
```

**UI Components:**
- `CutoutPanel.svelte`
- `QualityPanel.svelte`
- `SettingsPanel.svelte`
- `Tabs.svelte` (navigation)

**Success Criteria:**
- All three tools functional
- Consistent UI patterns across panels
- Settings persist (store in JSON file)

---

### Phase 3: Batch Processing (Week 5-6)

**Goal:** Process multiple images efficiently

**Features:**
- ✅ Batch queue management
- ✅ Progress tracking per image
- ✅ Parallel processing (configurable workers)
- ✅ Export all results

**New Tauri Commands:**
```rust
// src-tauri/src/commands/batch.rs
#[tauri::command]
async fn process_batch(
    images: Vec<String>,
    operation: BatchOperation,
    config: serde_json::Value,
) -> Result<BatchProgress, String>;

// Emit events for progress updates
app_handle.emit("batch-progress", BatchProgressEvent {
    current: 5,
    total: 50,
    current_file: "robin.jpg",
});
```

**UI Components:**
- `BatchPanel.svelte`
- `BatchQueue.svelte`
- `ProgressBar.svelte`

**Success Criteria:**
- Can queue 50+ images
- Progress updates in real-time
- Can cancel batch mid-process
- Average processing time meets CLI benchmarks

---

### Phase 4: Polish & Advanced (Week 7-8)

**Goal:** Production-ready refinements

**Features:**
- ✅ Drag-and-drop image loading
- ✅ Keyboard shortcuts (Cmd/Ctrl+O for open, etc.)
- ✅ Recent files list
- ✅ Export presets
- ✅ Dark mode toggle
- ✅ Metadata viewer (TOML/JSON)
- ✅ Side-by-side comparison view

**Success Criteria:**
- App feels polished and responsive
- No major UX friction points
- Ready for alpha testing

---

## 7. Performance Considerations

### 7.1 Async Processing

**Problem:** ONNX inference blocks main thread

**Solution:** Use Tauri's async commands with tokio

```rust
#[tauri::command]
async fn detect_objects(request: DetectRequest) -> Result<DetectResponse, String> {
    // Spawn blocking task for ONNX inference
    tokio::task::spawn_blocking(move || {
        let pipeline = DetectionPipeline::new(config)?;
        pipeline.process_image(&request.image_path)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}
```

### 7.2 Asset Protocol for Images

**Always use `convertFileSrc`** for loading images:

```typescript
import { convertFileSrc } from '@tauri-apps/api/core';

// ✅ Good: Use asset protocol
const imageUrl = convertFileSrc('/path/to/crop.jpg');
<img src={imageUrl} alt="Crop" />

// ❌ Bad: Send base64 over IPC
const base64 = await invoke('get_image_base64', { path });
<img src={`data:image/jpeg;base64,${base64}`} alt="Crop" />
```

### 7.3 Debouncing Parameter Changes

**Problem:** Slider changes trigger too many detections

**Solution:** Debounce slider input

```typescript
import { debounce } from '$lib/utils/debounce';

const debouncedDetect = debounce(() => {
  runDetection();
}, 300); // 300ms delay

// In slider onchange
$effect(() => {
  if (autoDetect) {
    debouncedDetect();
  }
});
```

---

## 8. Code Snippets: Complete Examples

### 8.1 Main Tauri App Entry

**File:** `src-tauri/src/main.rs`

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod error;
mod state;

use commands::{detection, cutout, quality, filesystem};

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize app state
            let state = state::AppState::new();
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            detection::detect_objects,
            cutout::remove_background,
            quality::assess_quality,
            filesystem::select_output_directory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 8.2 Root Svelte Component

**File:** `src/App.svelte`

```svelte
<script lang="ts">
  import { Tabs, TabsList, TabsTrigger, TabsContent } from '$lib/components/ui/tabs';
  import DetectionPanel from '$lib/components/DetectionPanel.svelte';
  import CutoutPanel from '$lib/components/CutoutPanel.svelte';
  import QualityPanel from '$lib/components/QualityPanel.svelte';
  import SettingsPanel from '$lib/components/SettingsPanel.svelte';

  let activeTab = $state('detection');
</script>

<main class="h-screen flex flex-col">
  <!-- Header -->
  <header class="border-b p-4">
    <h1 class="text-2xl font-bold">Beaker - Bird Image Analysis</h1>
  </header>

  <!-- Main Content -->
  <div class="flex-1 overflow-auto">
    <Tabs bind:value={activeTab} class="h-full">
      <TabsList class="border-b">
        <TabsTrigger value="detection">Detection</TabsTrigger>
        <TabsTrigger value="cutout">Background Removal</TabsTrigger>
        <TabsTrigger value="quality">Quality Assessment</TabsTrigger>
        <TabsTrigger value="settings">Settings</TabsTrigger>
      </TabsList>

      <TabsContent value="detection">
        <DetectionPanel />
      </TabsContent>

      <TabsContent value="cutout">
        <CutoutPanel />
      </TabsContent>

      <TabsContent value="quality">
        <QualityPanel />
      </TabsContent>

      <TabsContent value="settings">
        <SettingsPanel />
      </TabsContent>
    </Tabs>
  </div>
</main>
```

### 8.3 Vite Configuration

**File:** `vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    watch: {
      ignored: ['**/src-tauri/**'],
    },
  },
});
```

---

## 9. Development Workflow

### 9.1 Initial Setup

```bash
# 1. Create Tauri app
cd beaker
npm create tauri-app@latest -- --name beaker-gui --template svelte-ts

# 2. Install dependencies
cd beaker-gui
npm install

# 3. Install UI components
npx shadcn-svelte@latest init
npx shadcn-svelte@latest add button slider card input label tabs

# 4. Install Tailwind
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 5. Configure workspace
# Edit root Cargo.toml to add beaker-gui/src-tauri to workspace
```

### 9.2 Development Loop

```bash
# Terminal 1: Run Tauri dev server
cd beaker-gui
npm run tauri dev

# Terminal 2: Watch for Rust changes
cd beaker-gui/src-tauri
cargo watch -x build

# Hot reload:
# - Frontend changes: Vite handles automatically
# - Rust backend changes: Requires app restart (or use cargo-watch)
```

### 9.3 Building for Production

```bash
# Build optimized bundle
npm run tauri build

# Outputs:
# - macOS: beaker-gui/src-tauri/target/release/bundle/dmg/Beaker_0.1.0_aarch64.dmg
# - Linux: beaker-gui/src-tauri/target/release/bundle/deb/beaker-gui_0.1.0_amd64.deb
# - Windows: beaker-gui/src-tauri/target/release/bundle/msi/Beaker_0.1.0_x64_en-US.msi
```

---

## 10. Cookie-Cutter Template: Adding a New Feature

### Example: Adding Species Classification

**Scenario:** Beaker lib adds `species` module that classifies bird species.

**Step-by-Step:**

#### 10.1 Rust Backend (15 minutes)

**File:** `src-tauri/src/commands/species.rs`

```rust
use beaker::core::species::{SpeciesClassifier, SpeciesConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub image_path: String,
    pub top_k: usize,
}

#[derive(Debug, Serialize)]
pub struct ClassifyResponse {
    pub predictions: Vec<SpeciesPrediction>,
    pub processing_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct SpeciesPrediction {
    pub species_name: String,
    pub confidence: f32,
}

#[tauri::command]
pub async fn classify_species(
    request: ClassifyRequest,
) -> Result<ClassifyResponse, String> {
    tokio::task::spawn_blocking(move || {
        let classifier = SpeciesClassifier::new(SpeciesConfig {
            top_k: request.top_k,
            ..Default::default()
        })?;

        let result = classifier.classify(&request.image_path)?;

        Ok(ClassifyResponse {
            predictions: result.predictions.into_iter().map(|p| SpeciesPrediction {
                species_name: p.species,
                confidence: p.confidence,
            }).collect(),
            processing_time_ms: result.processing_time_ms,
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
```

**Register in main.rs:**
```rust
use commands::species;

.invoke_handler(tauri::generate_handler![
    // ... existing
    species::classify_species,
])
```

#### 10.2 TypeScript Types (5 minutes)

**File:** `src/lib/types/beaker.ts`

```typescript
export interface ClassifyRequest {
  imagePath: string;
  topK: number;
}

export interface ClassifyResponse {
  predictions: SpeciesPrediction[];
  processingTimeMs: number;
}

export interface SpeciesPrediction {
  speciesName: string;
  confidence: number;
}
```

#### 10.3 API Wrapper (5 minutes)

**File:** `src/lib/api/species.ts`

```typescript
import { invoke } from '@tauri-apps/api/core';
import type { ClassifyRequest, ClassifyResponse } from '$lib/types/beaker';

export const speciesApi = {
  async classify(request: ClassifyRequest): Promise<ClassifyResponse> {
    return await invoke<ClassifyResponse>('classify_species', { request });
  },
};
```

#### 10.4 Svelte Store (10 minutes)

**File:** `src/lib/stores/species.ts`

```typescript
import { writable, derived } from 'svelte/store';
import type { ClassifyResponse } from '$lib/types/beaker';

export const speciesParams = writable({
  topK: 5,
});

export const currentImage = writable<string | null>(null);
export const classifyResults = writable<ClassifyResponse | null>(null);
export const isClassifying = writable(false);
export const classifyError = writable<string | null>(null);

export const topPrediction = derived(
  classifyResults,
  ($results) => $results?.predictions[0] ?? null
);
```

#### 10.5 UI Component (30 minutes)

**File:** `src/lib/components/SpeciesPanel.svelte`

```svelte
<script lang="ts">
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { open } from '@tauri-apps/plugin-dialog';
  import { convertFileSrc } from '@tauri-apps/api/core';
  import { speciesApi } from '$lib/api/species';
  import {
    speciesParams,
    currentImage,
    classifyResults,
    isClassifying,
    classifyError,
    topPrediction,
  } from '$lib/stores/species';

  let topK = $state($speciesParams.topK);

  async function selectImage() {
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }],
    });
    if (selected) currentImage.set(selected);
  }

  async function runClassification() {
    if (!$currentImage) return;

    isClassifying.set(true);
    classifyError.set(null);

    try {
      const result = await speciesApi.classify({
        imagePath: $currentImage,
        topK,
      });
      classifyResults.set(result);
    } catch (err) {
      classifyError.set(err instanceof Error ? err.message : String(err));
    } finally {
      isClassifying.set(false);
    }
  }

  $effect(() => {
    speciesParams.set({ topK });
  });
</script>

<div class="grid grid-cols-2 gap-4 p-4">
  <Card>
    <CardHeader><CardTitle>Image</CardTitle></CardHeader>
    <CardContent>
      {#if $currentImage}
        <img src={convertFileSrc($currentImage)} alt="Selected" class="w-full rounded-lg" />
      {:else}
        <div class="h-64 bg-muted rounded-lg flex items-center justify-center">
          <p class="text-muted-foreground">No image selected</p>
        </div>
      {/if}
    </CardContent>
  </Card>

  <Card>
    <CardHeader><CardTitle>Species Classification</CardTitle></CardHeader>
    <CardContent class="space-y-4">
      <Button onclick={selectImage} class="w-full">Select Image</Button>
      <Button onclick={runClassification} disabled={!$currentImage || $isClassifying} class="w-full">
        {$isClassifying ? 'Classifying...' : 'Classify Species'}
      </Button>

      {#if $classifyResults}
        <div class="space-y-2">
          <h3 class="font-semibold">Top Predictions:</h3>
          {#each $classifyResults.predictions as pred}
            <div class="flex justify-between">
              <span class="capitalize">{pred.speciesName}</span>
              <span>{(pred.confidence * 100).toFixed(1)}%</span>
            </div>
          {/each}
        </div>
      {/if}

      {#if $classifyError}
        <p class="text-sm text-destructive">{$classifyError}</p>
      {/if}
    </CardContent>
  </Card>
</div>
```

#### 10.6 Wire Up in App.svelte (2 minutes)

```svelte
<script lang="ts">
  import SpeciesPanel from '$lib/components/SpeciesPanel.svelte';
</script>

<TabsList class="border-b">
  <!-- ... existing tabs -->
  <TabsTrigger value="species">Species</TabsTrigger>
</TabsList>

<TabsContent value="species">
  <SpeciesPanel />
</TabsContent>
```

**Total time:** ~1 hour for a complete new feature!

---

## 11. Next Steps

### Immediate Actions (Week 1)

1. **Setup project structure**
   ```bash
   cd beaker
   npm create tauri-app@latest
   # Follow prompts, select Svelte + TypeScript
   ```

2. **Configure workspace**
   - Add `beaker-gui/src-tauri` to root `Cargo.toml` workspace members
   - Verify `beaker` lib dependency in `beaker-gui/src-tauri/Cargo.toml`

3. **Install UI dependencies**
   ```bash
   cd beaker-gui
   npm install -D tailwindcss postcss autoprefixer
   npx shadcn-svelte@latest init
   ```

4. **Implement Phase 1 MVP**
   - Detection command + panel
   - Basic styling
   - Asset protocol working

### Success Metrics

**Week 2 (MVP):**
- ✅ Can select image, run detection, view crops
- ✅ Processing time <100ms per image
- ✅ UI looks polished, not prototypey

**Week 4 (Core Features):**
- ✅ All three tools (detection, cutout, quality) working
- ✅ Consistent UI patterns
- ✅ Settings panel functional

**Week 6 (Batch):**
- ✅ Can process 50+ images in batch
- ✅ Real-time progress updates
- ✅ Performance matches CLI benchmarks

**Week 8 (Polish):**
- ✅ Drag-and-drop, keyboard shortcuts, dark mode
- ✅ Ready for alpha testing

---

## 12. Testing Strategy (CLI-First, No Browser Automation)

### 12.1 Testing Philosophy

**Goal:** Comprehensive testing via CLI commands that agents can run. Avoid complex browser automation.

**Strategy:**
1. **Rust Backend Tests** (70% coverage) - Test Tauri commands as pure functions
2. **Frontend Unit Tests** (20% coverage) - Test stores, utils, API wrappers with Vitest
3. **Integration Tests** (10% coverage) - Leverage existing beaker lib tests
4. **Manual E2E** (smoke tests only) - Quick manual verification, not automated

**What NOT to test:**
- ❌ Visual appearance (CSS/styling) - Manual QA only
- ❌ Complex UI interactions - Trust Svelte/shadcn-svelte
- ❌ Browser-specific quirks - Manual testing on target platforms
- ❌ Beaker lib logic - Already tested in beaker crate

---

### 12.2 Rust Backend Tests (Primary Test Layer)

**Key Insight:** Tauri commands are just Rust functions - test them directly!

#### 12.2.1 Command Unit Tests

**File:** `src-tauri/src/commands/detection.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_detect_objects_success() {
        let request = DetectRequest {
            image_path: "tests/fixtures/sparrow.jpg".to_string(),
            confidence: 0.5,
            classes: vec!["bird".to_string(), "head".to_string()],
        };

        let result = detect_objects(request).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.detections.is_empty());
        assert!(response.processing_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_detect_objects_invalid_path() {
        let request = DetectRequest {
            image_path: "/nonexistent/image.jpg".to_string(),
            confidence: 0.5,
            classes: vec!["bird".to_string()],
        };

        let result = detect_objects(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_detect_objects_low_confidence() {
        let request = DetectRequest {
            image_path: "tests/fixtures/sparrow.jpg".to_string(),
            confidence: 0.95, // Very high threshold
            classes: vec!["bird".to_string()],
        };

        let result = detect_objects(request).await;
        assert!(result.is_ok());
        // May have fewer or zero detections
    }

    #[test]
    fn test_bbox_serialization() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 150.0,
        };

        let json = serde_json::to_string(&bbox).unwrap();
        assert!(json.contains("\"x\":10.0"));
        assert!(json.contains("\"y\":20.0"));
    }
}
```

#### 12.2.2 Test Fixtures

**Structure:**
```
src-tauri/
├── tests/
│   ├── fixtures/
│   │   ├── sparrow.jpg         # Known good image
│   │   ├── cardinal.jpg        # Another test image
│   │   ├── no_bird.jpg         # Image with no birds
│   │   └── corrupted.jpg       # Invalid image
│   ├── integration_tests.rs    # Integration tests
│   └── common/
│       └── mod.rs              # Test helpers
```

**Test Helpers:**

```rust
// src-tauri/tests/common/mod.rs
use std::path::PathBuf;

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

pub fn assert_valid_detection(response: &DetectResponse) {
    assert!(!response.detections.is_empty());
    assert!(response.processing_time_ms > 0.0);

    for detection in &response.detections {
        assert!(detection.confidence >= 0.0 && detection.confidence <= 1.0);
        assert!(detection.bbox.width > 0.0);
        assert!(detection.bbox.height > 0.0);
    }
}
```

#### 12.2.3 Integration Tests

**File:** `src-tauri/tests/integration_tests.rs`

```rust
use beaker_gui::commands::{detection, cutout, quality};

#[tokio::test]
async fn test_full_detection_workflow() {
    let fixture = common::fixture_path("sparrow.jpg");

    // Test detection
    let detect_response = detection::detect_objects(detection::DetectRequest {
        image_path: fixture.to_string_lossy().to_string(),
        confidence: 0.5,
        classes: vec!["bird".to_string()],
    })
    .await
    .expect("Detection failed");

    common::assert_valid_detection(&detect_response);
    assert!(!detect_response.output_paths.is_empty());

    // Verify output files exist
    for path in &detect_response.output_paths {
        assert!(std::path::Path::new(path).exists(), "Crop file not created");
    }
}

#[tokio::test]
async fn test_cutout_workflow() {
    let fixture = common::fixture_path("cardinal.jpg");

    let cutout_response = cutout::remove_background(cutout::CutoutRequest {
        image_path: fixture.to_string_lossy().to_string(),
        post_process: true,
        alpha_matting: false,
        save_mask: false,
    })
    .await
    .expect("Cutout failed");

    assert!(!cutout_response.output_path.is_empty());
    assert!(std::path::Path::new(&cutout_response.output_path).exists());
}
```

#### 12.2.4 Running Rust Tests

```bash
# Run all backend tests
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml

# Run with output
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml -- --nocapture

# Run specific test
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml test_detect_objects_success

# Run integration tests only
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml --test integration_tests
```

---

### 12.3 Frontend Unit Tests (Vitest)

**Setup:**

```json
// package.json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "coverage": "vitest --coverage"
  },
  "devDependencies": {
    "vitest": "^2.0.0",
    "@testing-library/svelte": "^5.0.0",
    "@vitest/ui": "^2.0.0",
    "jsdom": "^25.0.0"
  }
}
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/tests/setup.ts'],
  },
});
```

#### 12.3.1 Store Tests

**File:** `src/lib/stores/detection.test.ts`

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import {
  detectionParams,
  detectionResults,
  isDetecting,
  processingTime,
  detectionCount,
} from './detection';

describe('detection store', () => {
  beforeEach(() => {
    // Reset stores
    detectionParams.set({ confidence: 0.5, classes: ['bird', 'head'] });
    detectionResults.set(null);
    isDetecting.set(false);
  });

  it('should initialize with default params', () => {
    const params = get(detectionParams);
    expect(params.confidence).toBe(0.5);
    expect(params.classes).toEqual(['bird', 'head']);
  });

  it('should update detection params', () => {
    detectionParams.set({ confidence: 0.8, classes: ['bird'] });
    const params = get(detectionParams);
    expect(params.confidence).toBe(0.8);
    expect(params.classes).toEqual(['bird']);
  });

  it('should compute processing time from results', () => {
    detectionResults.set({
      detections: [],
      outputPaths: [],
      processingTimeMs: 123.45,
    });

    expect(get(processingTime)).toBe('123ms');
  });

  it('should compute detection count', () => {
    detectionResults.set({
      detections: [
        { className: 'bird', confidence: 0.9, bbox: { x: 0, y: 0, width: 10, height: 10 } },
        { className: 'head', confidence: 0.8, bbox: { x: 0, y: 0, width: 5, height: 5 } },
      ],
      outputPaths: [],
      processingTimeMs: 100,
    });

    expect(get(detectionCount)).toBe(2);
  });

  it('should return null when no results', () => {
    expect(get(processingTime)).toBeNull();
    expect(get(detectionCount)).toBe(0);
  });
});
```

#### 12.3.2 API Wrapper Tests (with Mocks)

**File:** `src/lib/api/detection.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { detectionApi, detectBirds } from './detection';

// Mock Tauri invoke
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

import { invoke } from '@tauri-apps/api/core';

describe('detection API', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should call detect with correct parameters', async () => {
    const mockResponse = {
      detections: [],
      outputPaths: [],
      processingTimeMs: 100,
    };

    vi.mocked(invoke).mockResolvedValue(mockResponse);

    const result = await detectionApi.detect({
      imagePath: '/path/to/image.jpg',
      confidence: 0.5,
      classes: ['bird'],
    });

    expect(invoke).toHaveBeenCalledWith('detect_objects', {
      request: {
        imagePath: '/path/to/image.jpg',
        confidence: 0.5,
        classes: ['bird'],
      },
    });

    expect(result).toEqual(mockResponse);
  });

  it('should use default parameters in detectBirds', async () => {
    const mockResponse = {
      detections: [],
      outputPaths: [],
      processingTimeMs: 100,
    };

    vi.mocked(invoke).mockResolvedValue(mockResponse);

    await detectBirds('/path/to/image.jpg');

    expect(invoke).toHaveBeenCalledWith('detect_objects', {
      request: {
        imagePath: '/path/to/image.jpg',
        confidence: 0.5,
        classes: ['bird', 'head'],
      },
    });
  });

  it('should handle errors', async () => {
    vi.mocked(invoke).mockRejectedValue(new Error('Detection failed'));

    await expect(detectionApi.detect({
      imagePath: '/invalid/path.jpg',
      confidence: 0.5,
      classes: ['bird'],
    })).rejects.toThrow('Detection failed');
  });
});
```

#### 12.3.3 Utility Tests

**File:** `src/lib/utils/formatters.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { formatProcessingTime, formatFileSize } from './formatters';

describe('formatters', () => {
  describe('formatProcessingTime', () => {
    it('should format milliseconds', () => {
      expect(formatProcessingTime(123.45)).toBe('123ms');
      expect(formatProcessingTime(1234.56)).toBe('1.23s');
      expect(formatProcessingTime(50.1)).toBe('50ms');
    });
  });

  describe('formatFileSize', () => {
    it('should format bytes', () => {
      expect(formatFileSize(500)).toBe('500 B');
      expect(formatFileSize(1024)).toBe('1.0 KB');
      expect(formatFileSize(1048576)).toBe('1.0 MB');
      expect(formatFileSize(2.5 * 1024 * 1024)).toBe('2.5 MB');
    });
  });
});
```

#### 12.3.4 Running Frontend Tests

```bash
# Run all frontend tests
npm test

# Run with coverage
npm run coverage

# Run in watch mode
npm test -- --watch

# Run specific test file
npm test -- detection.test.ts

# Run with UI (interactive)
npm run test:ui
```

---

### 12.4 Test Coverage Goals

**Backend (Rust):**
- Command functions: **90%+** coverage
- Request/response types: **100%** (serialization tests)
- Error handling: **80%+** coverage
- State management: **70%+** coverage

**Frontend (TypeScript):**
- Stores: **80%+** coverage
- API wrappers: **90%+** coverage (mostly mocked)
- Utilities: **90%+** coverage
- Components: **30-50%** coverage (focus on logic, not rendering)

**Overall:** Aim for **70%+** total coverage across backend + frontend.

---

### 12.5 CI/CD Integration

**GitHub Actions Workflow:**

```yaml
# .github/workflows/gui-tests.yml
name: GUI Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    paths:
      - 'beaker-gui/**'
      - 'beaker/src/**'

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run backend tests
        run: cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml

      - name: Run backend tests with coverage
        run: |
          cargo install cargo-tarpaulin
          cargo tarpaulin --manifest-path=beaker-gui/src-tauri/Cargo.toml --out Xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: beaker-gui/package-lock.json

      - name: Install dependencies
        working-directory: beaker-gui
        run: npm ci

      - name: Run frontend tests
        working-directory: beaker-gui
        run: npm test

      - name: Run frontend tests with coverage
        working-directory: beaker-gui
        run: npm run coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

---

### 12.6 Manual Testing Checklist (Smoke Tests)

**Run before each release:**

- [ ] App launches successfully
- [ ] Can select image via file dialog
- [ ] Detection runs and displays crops
- [ ] Cutout runs and shows result
- [ ] Quality assessment runs
- [ ] Settings can be changed and persist
- [ ] Batch processing works for 10 images
- [ ] Dark mode toggle works
- [ ] Asset protocol loads images correctly
- [ ] Error messages display properly
- [ ] App can be closed cleanly

**Estimated time:** 15-20 minutes

---

### 12.7 Testing Command Reference

```bash
# Backend tests (Rust)
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml

# Frontend tests (Vitest)
cd beaker-gui && npm test

# All tests (both)
./scripts/test-all.sh  # Create this script

# Coverage
cargo tarpaulin --manifest-path=beaker-gui/src-tauri/Cargo.toml
npm run coverage --prefix beaker-gui

# Continuous (watch mode)
cargo watch -x test --manifest-path=beaker-gui/src-tauri/Cargo.toml
npm test -- --watch --prefix beaker-gui
```

**Create `scripts/test-all.sh`:**

```bash
#!/bin/bash
set -e

echo "Running backend tests..."
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml

echo "Running frontend tests..."
cd beaker-gui && npm test

echo "All tests passed! ✅"
```

---

### 12.8 What About E2E Tests?

**Decision: Skip automated E2E for now**

**Rationale:**
- Complex setup (WebDriver, browser automation)
- Brittle (breaks with UI changes)
- Slow to run
- Most logic already tested in backend/frontend unit tests

**Alternative:**
- Rely on manual smoke tests (15 min checklist)
- Focus testing effort on backend commands (where bugs are costly)
- Use Vitest for frontend logic
- Trust shadcn-svelte components (already well-tested)

**If E2E becomes necessary:**
- Consider [Tauri WebDriver](https://tauri.app/v1/guides/testing/webdriver/introduction/)
- Or [Playwright for Tauri](https://github.com/spacedriveapp/spacedrive/tree/main/apps/desktop/src-tauri/tests)
- Keep tests minimal (critical path only)

---

## 13. Appendix: Quick Reference

### Build Commands

```bash
# Dev mode (hot reload)
npm run tauri dev

# Build production
npm run tauri build

# Run tests
cargo test --manifest-path=beaker-gui/src-tauri/Cargo.toml
```

### Key Files

| File | Purpose |
|------|---------|
| `src-tauri/tauri.conf.json` | Tauri configuration |
| `src-tauri/src/main.rs` | Rust entry point |
| `src-tauri/src/commands/` | Tauri command handlers |
| `src/App.svelte` | Root UI component |
| `src/lib/api/` | TypeScript API wrappers |
| `src/lib/stores/` | Svelte stores (state) |
| `src/lib/components/` | UI components |

### Asset Protocol Pattern

```typescript
// Always use convertFileSrc for file paths
import { convertFileSrc } from '@tauri-apps/api/core';

const url = convertFileSrc('/absolute/path/to/image.jpg');
<img src={url} alt="Image" />
```

### Adding New Command Checklist

- [ ] Create `src-tauri/src/commands/feature.rs`
- [ ] Define request/response types with Serialize/Deserialize
- [ ] Implement `#[tauri::command]` function
- [ ] Register in `main.rs` `invoke_handler`
- [ ] Add TypeScript types in `src/lib/types/beaker.ts`
- [ ] Create API wrapper in `src/lib/api/feature.ts`
- [ ] Create Svelte store in `src/lib/stores/feature.ts`
- [ ] Build UI component in `src/lib/components/FeaturePanel.svelte`
- [ ] Wire up in `App.svelte`

---

**Document Version:** 1.0
**Last Updated:** 2025-10-25
