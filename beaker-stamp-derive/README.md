# beaker-stamp-derive

Procedural macro for automatic stamp generation in Beaker configuration structs.

This crate provides the `#[derive(Stamp)]` macro that automatically implements the `Stamp` trait for configuration structs. Fields marked with `#[stamp]` are included in the deterministic hash, while unmarked fields are ignored.

## Features

- **Derive macro**: Automatically implement `Stamp` trait
- **Field selection**: Use `#[stamp]` attribute to mark fields affecting output bytes
- **Attribute options**: Support for field renaming and custom serialization functions
- **Compile-time validation**: Ensures only supported field types are stamped

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

## Supported Attributes

### Basic stamping
```rust
#[stamp]
pub field: Type,
```

### Field renaming
```rust
#[stamp(rename = "custom_name")]
pub field: Type,
```

### Custom serialization
```rust
#[stamp(with = "custom_serialize_fn")]
pub field: Type,
```

The custom function should have signature `fn(&T) -> impl Serialize`.

## Generated Implementation

The macro generates a `stamp_value()` method that creates a JSON object containing only the stamped fields:

```rust
impl Stamp for DetectionConfig {
    fn stamp_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("confidence".to_string(), serde_json::to_value(&self.confidence).unwrap());
        map.insert("crop_classes".to_string(), serde_json::to_value(&self.crop_classes).unwrap());
        serde_json::Value::Object(map)
    }
}
```

## Design Philosophy

The macro enforces a clear separation between configuration parameters that affect byte-level output (stamped) and those that only affect performance or metadata (not stamped). This ensures that incremental builds trigger only when actual processing results would change.

## Error Handling

- Compile-time errors for unsupported syntax (unnamed fields, enums, unions)
- Runtime panics for serialization failures (typically only with custom types)
- Clear error messages directing users to the `#[stamp]` attribute
