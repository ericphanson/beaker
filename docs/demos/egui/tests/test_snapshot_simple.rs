// Simple test to check if wgpu snapshot generation works
use egui_kittest::Harness;

#[test]
fn test_snapshot_works() {
    let mut harness = Harness::new_ui(|ui| {
        ui.label("Test");
    });

    harness.run();

    // Try to create a snapshot - this will panic if it doesn't work
    harness.wgpu_snapshot("test_snapshot");
}
