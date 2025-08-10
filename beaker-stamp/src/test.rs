use beaker_stamp::Stamp;
use beaker_stamp_derive::Stamp as DeriveStamp;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, DeriveStamp)]
pub struct TestConfig {
    #[stamp]
    pub important_value: u32,
    #[stamp]
    pub another_value: String,
    pub ignored_value: bool, // Not stamped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stamp_generation() {
        let config = TestConfig {
            important_value: 42,
            another_value: "test".to_string(),
            ignored_value: true,
        };

        let stamp_value = config.stamp_value();
        println!("Stamp value: {}", stamp_value);

        let hash = config.stamp_hash();
        println!("Stamp hash: {}", hash);

        assert!(hash.starts_with("sha256:"));
    }

    #[test]
    fn test_stamp_determinism() {
        let config1 = TestConfig {
            important_value: 42,
            another_value: "test".to_string(),
            ignored_value: true,
        };

        let config2 = TestConfig {
            important_value: 42,
            another_value: "test".to_string(),
            ignored_value: false, // Different ignored value
        };

        // Should be same since ignored_value is not stamped
        assert_eq!(config1.stamp_hash(), config2.stamp_hash());
    }

    #[test]
    fn test_stamp_changes_with_important_fields() {
        let config1 = TestConfig {
            important_value: 42,
            another_value: "test".to_string(),
            ignored_value: true,
        };

        let config2 = TestConfig {
            important_value: 43, // Different important value
            another_value: "test".to_string(),
            ignored_value: true,
        };

        // Should be different since important_value changed
        assert_ne!(config1.stamp_hash(), config2.stamp_hash());
    }
}
