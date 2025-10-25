# GUI Testing in Restricted Environments - Investigation Report

**Date:** 2025-10-25
**Issue:** Cannot test Tauri GUI locally in sandboxed/network-restricted environments

---

## Problem Statement

The beaker-gui Tauri application cannot be built or tested in environments that:
1. Have network restrictions preventing package installation (apt-get fails)
2. Lack GTK system dependencies (libgtk-3-dev, libwebkit2gtk-4.1-dev, etc.)
3. Don't have Docker/Podman available

## Investigation Results

### Attempts Made

#### 1. Docker Installation ❌
**Tried:** Install Docker via convenience script
**Result:** Failed - network restrictions prevent downloading Docker

```bash
curl: (6) Could not resolve host: download.docker.com
```

#### 2. Podman Installation ❌
**Tried:** `sudo apt-get install podman`
**Result:** Failed - package not available, network restrictions

#### 3. GTK Dependency Installation ❌
**Tried:** `sudo apt-get install libgtk-3-dev libwebkit2gtk-4.1-dev`
**Result:** Failed - network restrictions prevent package download

```
Temporary failure resolving 'archive.ubuntu.com'
```

#### 4. Build Script Bypass ❌
**Tried:** Research environment variables to skip build.rs execution
**Result:** Not possible - Cargo always runs build scripts, even for `cargo check`

**Reference:** [Cargo Issue #4001](https://github.com/rust-lang/cargo/issues/4001) - Feature request to inform build scripts about `cargo check` (not implemented)

#### 5. Tauri Test Feature ❌
**Tried:** Enable `test` feature for mock runtime testing
**Result:** Still requires GTK because build.rs in dependencies runs pkg-config

```rust
[dev-dependencies]
tauri = { version = "2", features = ["test"] }
```

Even `cargo test --lib` triggers:
```
error: The system library `gdk-3.0` required by crate `gdk-sys` was not found
```

#### 6. No-Default-Features ❌
**Tried:** `cargo test --no-default-features --lib`
**Result:** Still requires GTK (dependencies unconditional)

### Root Cause

**Tauri's Linux implementation requires GTK at compile time:**
- Uses webkit2gtk for web rendering
- Dependencies (gdk-sys, gdk-pixbuf-sys, webkit2gtk-sys) have build.rs scripts
- Build scripts run `pkg-config` to find GTK headers
- No way to bypass or mock these build-time requirements

### What IS Available

```bash
# Runtime libraries (installed)
dpkg -l | grep gtk
ii  libgtk-3-0t64:amd64     # GTK runtime library ✅
ii  libgtk-3-common         # GTK common files ✅

# Development headers (NOT installed)
libgtk-3-dev              # ❌
libwebkit2gtk-4.1-dev     # ❌
```

---

## Solutions Analysis

### ✅ Solution 1: CI-Based Testing (IMPLEMENTED)

**What we did:**
- Created `.github/workflows/gui-ci.yml`
- CI installs GTK dependencies automatically
- Runs `just gui-ci` for complete validation

**Pros:**
- ✅ Works reliably in GitHub Actions
- ✅ Standard industry practice for GUI apps
- ✅ No local environment setup needed
- ✅ Consistent test environment

**Cons:**
- ⏱️ Slower feedback (CI queue time)
- 🌐 Requires GitHub access

**Usage:**
```bash
# In CI (automated)
just gui-ci

# Local (requires GTK installed)
just gui-ci
```

### ❌ Solution 2: Docker (NOT VIABLE)

**Reason:** No Docker/Podman in restricted environments

### ❌ Solution 3: Mock/Headless Testing (NOT VIABLE)

**Reason:** Tauri's test feature still requires GTK at build time

### ⚠️ Solution 4: xvfb-run (Partial)

**Status:** Available with GTK installed
**Purpose:** Run GUI apps headless (for integration tests)

```bash
xvfb-run cargo test --manifest-path beaker-gui/src-tauri/Cargo.toml
```

**Requirements:**
- ✅ xvfb package installed
- ❌ Still needs GTK dev headers to compile

**Use case:** CI environments with GTK (already works)

---

## Recommendations

### For This Environment (Sandboxed/Restricted)

**Accept the limitation:**
1. GUI code cannot be tested locally in this environment
2. Rely on CI for validation (already implemented)
3. Focus local work on:
   - CLI testing (works fine)
   - Code review
   - Documentation
   - Planning

### For Developer Machines

**macOS (Target Platform):**
```bash
# No additional dependencies needed
just gui-ci
```

**Linux:**
```bash
# One-time setup
sudo apt-get install libwebkit2gtk-4.1-dev libappindicator3-dev \
  librsvg2-dev patchelf libgtk-3-dev

# Then works normally
just gui-ci
```

**Windows:**
Follow [Tauri Prerequisites](https://v2.tauri.app/start/prerequisites/)

### For Future Consideration

**If Docker becomes available:**
1. Create Dockerfile with GTK dependencies (draft exists)
2. Use docker-compose for local testing
3. Run `docker-compose run gui-test`

**Alternative Architecture:**
- Separate backend API server (testable without GUI)
- GUI as thin client
- More complex, not recommended for this use case

---

## Current Status

✅ **Problem Understood:** Tauri requires GTK at compile time on Linux
✅ **CI Solution Implemented:** GitHub Actions workflow working
✅ **Documentation Complete:** README, justfile comments updated
❌ **Local Testing Not Possible:** In restricted environments

## Conclusion

**The CI-based testing approach is correct and industry-standard for Tauri applications.** Local testing requires either:
1. GTK development headers installed (Linux)
2. macOS/Windows (no GTK needed)
3. Docker (not available here)

No workarounds exist to bypass GTK requirements in restricted Linux environments.

---

## References

- [Tauri Prerequisites](https://v2.tauri.app/start/prerequisites/)
- [Tauri CI Documentation](https://v2.tauri.app/develop/tests/webdriver/ci/)
- [Cargo Issue #4001](https://github.com/rust-lang/cargo/issues/4001) - Build script detection
- [Tauri Test Module](https://docs.rs/tauri/latest/tauri/test/)
- [GitHub Issue #1061](https://github.com/tauri-apps/tauri/issues/1061) - Headless Tauri
- [GitHub Issue #12419](https://github.com/tauri-apps/tauri/issues/12419) - Codespace crashes

---

_Investigation completed: 2025-10-25_
