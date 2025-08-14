# beaker-stamp

Core traits and utilities for stamp-based dependency tracking in Beaker.

This crate provides the foundation for generating deterministic configuration stamps that enable Make-compatible incremental builds. It implements canonical JSON serialization and stamp file management with atomic writes and mtime preservation.

## Features

- **Stamp trait**: Core trait for generating deterministic hashes from configuration objects
- **Canonical JSON**: Stable JSON serialization with sorted object keys and normalized numbers
- **Atomic writes**: Temp file + rename pattern with mtime preservation for unchanged content
- **Configurable directories**: Support for custom stamp directories via `BEAKER_STAMP_DIR`

## Usage

Implement the `Stamp` trait for configuration structs, typically using the derive macro from `beaker-stamp-derive`:

```rust
use beaker_stamp::Stamp;

#[derive(Stamp)]
struct Config {
    #[stamp] threshold: f32,     // Included in stamp hash
    #[stamp] enabled: bool,      // Included in stamp hash
    verbosity: u8,               // Not stamped - performance only
}
```

The trait provides deterministic hash generation:

```rust
let config = Config { threshold: 0.5, enabled: true, verbosity: 2 };
let hash = config.stamp_hash(); // "sha256:..."
```

Write stamp files to cache directory:

```rust
use beaker_stamp::write_cfg_stamp;

let stamp_path = write_cfg_stamp("detect", &config)?;
// Writes to ~/.cache/beaker/stamps/cfg-detect-<hash>.stamp
```

## Stamp File Format

Stamp files contain one line with the tool name and hash:

```
cfg=detect sha256:f5fc808f3ba2145d...
```

## Design Principles

- **Deterministic**: Same configuration always produces same hash
- **Minimal**: Only byte-affecting parameters included in stamps
- **Atomic**: Writes are atomic and preserve mtime when content unchanged
- **Portable**: Works across different platforms and Rust versions
