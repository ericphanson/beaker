use beaker::config::{BaseModelConfig, QualityConfig};
use beaker::quality_types::{ColorMap, QualityParams};

#[test]
fn test_quality_config_default_params() {
    let base = BaseModelConfig {
        sources: vec!["test.jpg".into()],
        device: "cpu".to_string(),
        output_dir: None,
        skip_metadata: true,
        strict: true,
        force: false,
    };

    let config = QualityConfig {
        base,
        model_path: None,
        model_url: None,
        model_checksum: None,
        debug_dump_images: false,
        params: None,
        heatmap_output: None,
        colormap: ColorMap::Viridis,
        overlay: false,
    };

    let params = config.params.unwrap_or_default();
    assert_eq!(params.alpha, 0.7);
    assert_eq!(params.beta, 1.2);
}

#[test]
fn test_quality_config_custom_params() {
    let custom_params = QualityParams {
        alpha: 0.8,
        beta: 1.5,
        ..Default::default()
    };

    let base = BaseModelConfig {
        sources: vec!["test.jpg".into()],
        device: "cpu".to_string(),
        output_dir: None,
        skip_metadata: true,
        strict: true,
        force: false,
    };

    let config = QualityConfig {
        base,
        model_path: None,
        model_url: None,
        model_checksum: None,
        debug_dump_images: false,
        params: Some(custom_params),
        heatmap_output: None,
        colormap: ColorMap::Viridis,
        overlay: false,
    };

    assert!(config.params.is_some());
    assert_eq!(config.params.as_ref().unwrap().alpha, 0.8);
    assert_eq!(config.params.as_ref().unwrap().beta, 1.5);
}
