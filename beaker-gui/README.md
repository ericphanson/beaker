# Beaker GUI

Tauri-based desktop GUI for Beaker bird detection toolkit.

## Prerequisites

### System Dependencies

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev \
  libappindicator3-dev \
  librsvg2-dev \
  patchelf \
  libgtk-3-dev
```

**macOS:**
No additional dependencies needed beyond Xcode Command Line Tools.

**Windows:**
Refer to [Tauri Prerequisites](https://v2.tauri.app/start/prerequisites/).

### Development Tools

- Rust (stable)
- Node.js 22+
- npm

## Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev

# Check types
npm run check

# Build for production
npm run tauri build
```

## Testing

```bash
# Run backend tests (requires GTK on Linux)
cargo test --manifest-path src-tauri/Cargo.toml

# Run all GUI CI checks (from repository root)
just gui-ci
```

## Architecture

See `docs/plans/2025-10-25-tauri-implementation-plan.md` for complete architecture and design decisions.

**Key Design:**
- Backend: Tauri 2.0 + async Rust commands
- Frontend: Svelte 5 + TypeScript + Tailwind CSS
- Images displayed via asset protocol (no IPC overhead)
- Heavy assertion-based testing

## CI

GUI CI runs on GitHub Actions with GTK dependencies pre-installed. Local development requires system dependencies listed above.

## Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Svelte](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer).
