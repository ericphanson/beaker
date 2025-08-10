# beaker-stamp-derive

Procedural macro for automatic stamp generation in Beaker configuration structs.

## What Are Procedural Macros?

Procedural macros (proc macros) are a powerful Rust feature that allow you to write code that generates other code at compile time. Unlike regular macros that work with syntax trees, proc macros operate on Rust tokens and can generate arbitrary Rust code.

There are three types of proc macros:
- **Derive macros** (`#[derive(MyTrait)]`) - automatically implement traits
- **Attribute macros** (`#[my_attribute]`) - transform the annotated item
- **Function-like macros** (`my_macro!(...)`) - similar to regular macros but more powerful

This crate implements a **derive macro** that automatically generates implementations of the `Stamp` trait.

## Why Is This in a Separate Crate?

Procedural macros in Rust **must** be defined in a separate crate with `proc-macro = true` in `Cargo.toml`. This is a Rust language requirement because:

1. **Compilation order**: Proc macros must be compiled before they can be used
2. **Different compilation target**: Proc macros run on the host machine during compilation, not in the target binary
3. **Dependency isolation**: Separates compile-time dependencies from runtime dependencies

## How This Proc Macro Works

### Input Processing
The macro receives the struct definition as a token stream:
```rust
#[derive(Stamp)]
pub struct DetectionConfig {
    #[stamp] pub confidence: f32,
    pub verbosity: u8,  // No #[stamp] attribute
}
```

### Code Generation
The macro analyzes each field and generates a `stamp_value()` method that:
1. Creates a JSON object containing only fields marked with `#[stamp]`
2. Serializes each stamped field using `serde_json`
3. Returns a deterministic JSON value for hashing

Generated code (conceptually):
```rust
impl Stamp for DetectionConfig {
    fn stamp_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("confidence".to_string(), serde_json::to_value(&self.confidence).unwrap());
        // Note: 'verbosity' is NOT included because it lacks #[stamp]
        serde_json::Value::Object(map)
    }
}
```

### Attribute Processing
The macro supports several attribute options:

**Basic stamping**:
```rust
#[stamp]
pub field: Type,
```

**Field renaming**:
```rust
#[stamp(rename = "custom_name")]
pub field: Type,
```

**Custom serialization**:
```rust
#[stamp(with = "custom_serialize_fn")]
pub field: Type,
```

The custom function should have signature `fn(&T) -> impl Serialize`.

## Features

- **Automatic trait implementation**: No manual `Stamp` trait implementations needed
- **Selective field inclusion**: Use `#[stamp]` attribute to mark fields affecting output bytes
- **Compile-time validation**: Ensures only supported field types are stamped
- **Clear error messages**: Helpful diagnostics for unsupported syntax

## Usage

```rust
use beaker_stamp_derive::Stamp;

#[derive(Stamp)]
pub struct DetectionConfig {
    #[stamp] pub confidence: f32,
    #[stamp] pub crop_classes: HashSet<DetectionClass>,
    pub verbosity: u8,  // Not stamped - performance setting only
}
```

## Design Philosophy

The macro enforces a clear separation between configuration parameters that affect byte-level output (stamped) and those that only affect performance or metadata (not stamped). This ensures that incremental builds trigger only when actual processing results would change.

## Error Handling

- **Compile-time errors** for unsupported syntax (unnamed fields, enums, unions)
- **Runtime panics** for serialization failures (typically only with custom types)
- **Clear error messages** directing users to the `#[stamp]` attribute

## Technical Implementation

The macro uses the `syn` crate to parse Rust syntax and `quote` to generate code. It:

1. Parses the input struct using `syn::DeriveInput`
2. Extracts fields marked with `#[stamp]` attributes
3. Generates serialization code for each stamped field
4. Outputs a complete `Stamp` trait implementation

This approach ensures compile-time safety while providing runtime efficiency.
