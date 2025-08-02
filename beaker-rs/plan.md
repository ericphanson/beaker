# 🦾 Cross-platform YOLO-Nano CLI – clean spec (no shims)

A condensed plan for an LLM-driven build agent to turn a **single Rust code-base** into **one portable binary per OS/arch**, accelerated by Core ML on macOS and CPU elsewhere, with full image pre-/post-processing.

---

## 1 Functional overview

| Layer | Required behaviour |
|-------|--------------------|
| **Input** | `--img path` or raw RGB via `stdin`. Accept `.jpg`/`.png`. |
| **Pre-processing** | 1 Load → RGB → `Vec<u8>`<br>2 **Letterbox-resize** to model size (e.g. `416×416`) with 114-gray padding<br>3 `u8 → f32`, scale `/255`<br>4 Reorder to **CHW** contiguous buffer<br>*(Optional flag to map [0,1]→[-1,1] if model trained that way)* |
| **Inference** | *macOS*: ONNX Runtime **Core ML Execution Provider** with CPU fallback<br>*Linux & Windows*: ONNX Runtime **CPU Execution Provider** only |
| **Post-processing** | 1 Decode YOLO head (anchor + stride tables)<br>2 Confidence = `sigmoid(obj)×sigmoid(cls)`<br>3 Filter `conf > 0.25`<br>4 Per-class **NMS** (IoU ≥ 0.45, top 300)<br>5 Return `Detection {x1,y1,x2,y2,score,class}` list |
| **Binary** | ≤ 20 MB, single file per target (mac slices dynamically link system frameworks). |
| **CLI UX** | `detect`, `benchmark`, `version`, `--save`, `--device cpu|auto`. |

---

## 2 Directory layout

```
.
├── Cargo.toml
├── build.rs                # adds CoreML / Metal frameworks on macOS
├── src/
│   ├── main.rs             # CLI & image I/O
│   ├── preprocess.rs       # letterbox + normalise
│   ├── tensor.rs           # CHW helpers
│   ├── decode.rs           # anchor/stride decode
│   ├── nms.rs              # IoU NMS (safe Rust)
│   └── inference.rs        # OrtSession wrapper & EP selection
└── model/
    └── yolo_nano.onnx      # embedded via include_bytes!
```

---

## 3 Cargo features / targets

```toml
[features]
default = ["cpu"]

cpu     = ["ort/cpu", "ort/compile-static"]    # always on
coreml  = ["cpu", "ort/coreml"]                # macOS targets only
```

| Target triple | Build flags | EPs compiled in | Notes |
|---------------|------------|-----------------|-------|
| `aarch64-apple-darwin` & `x86_64-apple-darwin` | `--features coreml` | Core ML ➜ CPU | `build.rs` links `CoreML`, `Metal`; set `MACOSX_DEPLOYMENT_TARGET=11.0`. |
| `x86_64-unknown-linux-musl` | `--features cpu` | CPU | Uses `cargo zigbuild` for fully-static musl ELF. |
| `x86_64-pc-windows-msvc` | `--features cpu` | CPU | Build with `/MT`; use `cargo xwin` for cross. |

---

## 4 Dependencies to add in `Cargo.toml`

| Crate | Purpose |
|-------|---------|
| `ort` (features above) | ONNX Runtime FFI wrapper. |
| `image`                | Decode/encode common formats. |
| `ndarray` + `ndarray-npy` *(optional)* | Easier tensor reshaping / debugging dumps. |
| `clap`                 | Modern CLI parsing. |
| `rayon` *(optional)*   | Parallel preprocessing batches. |
| `tracing` + `tracing-subscriber` | Structured logging / benchmarking. |

Everything else (NMS, decode, letterbox) is implemented in safe Rust.

---

## 5 Build & deployment gotchas

* **Core ML EP build time** – `--use_coreml` adds ~10–12 min to ORT static build; cache artefacts in CI.
* **macOS SDK** – Cross-compiling from Linux requires a mounted Xcode SDK; use `osxcross + cargo-zigbuild` or a macOS runner.
* **Universal binary** – After compiling both mac slices, run `lipo -create` to merge them.
* **Gatekeeper quarantine** – Unsigned CLI zips may get `com.apple.quarantine`; document `xattr -dr`.
* **Static musl quirks** – ORT’s static CPU build expects C++ 17; add `-static-libstdc++` arg if zig complains.
* **Model opset** – Verify exported ONNX opset ≤ the ORT you built (e.g. opset 14 for ORT 1.21 minimal).
* **Image alignment** – Letterbox ratios must match Ultralytics (scale to fit, then pad evenly) or detections drift.
* **SIMD** – For Linux release enable `RUSTFLAGS='-C target-cpu=native'`; for portable builds leave default.
* **Notarisation (optional)** – For wider Mac distribution, codesign + `notarytool altool`.

---

## 6 Release automation snippet (GitHub Actions)

```yaml
matrix:
  include:
    - {target: aarch64-apple-darwin, features: "coreml"}
    - {target: x86_64-apple-darwin,  features: "coreml"}
    - {target: x86_64-unknown-linux-musl, features: "cpu"}
    - {target: x86_64-pc-windows-msvc,   features: "cpu"}

steps:
  - uses: actions/checkout@v4
  - run: cargo install cargo-zigbuild cargo-xwin || true
  - run: ./ci/build.sh ${{ matrix.target }} ${{ matrix.features }}
  - run: strip target/${{ matrix.target }}/release/yolo-cli || true
  - run: zip -9 yolo_${{ matrix.target }}.zip target/.../yolo-cli
  - uses: softprops/action-gh-release@v2
    with: { files: yolo_${{ matrix.target }}.zip }
```

*(Inside `ci/build.sh` call either `cargo zigbuild` or `cargo xwin` as appropriate.)*

---

### ✅ Deliverables checklist for the agent

1. Static ORT builds (CPU everywhere, Core ML on mac) using minimal build flags.
2. Rust crate with preprocessing, decode, NMS in safe code.
3. Stripped binaries (≤ 20 MB) for four targets, zipped with checksums & release notes.

Follow these steps and the CLI will replicate Ultralytics’ behaviour while remaining dependency-free for end-users.
