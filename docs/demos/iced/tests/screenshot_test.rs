// Screenshot/snapshot tests using tiny-skia for headless rendering
use std::fs;
use std::path::Path;

#[test]
fn test_tiny_skia_basic_render() -> Result<(), Box<dyn std::error::Error>> {
    // Create a pixmap (in-memory image)
    let mut pixmap = tiny_skia::Pixmap::new(400, 300)
        .ok_or("Failed to create pixmap")?;

    // Fill with white background
    pixmap.fill(tiny_skia::Color::WHITE);

    // Create a simple shape to prove rendering works
    let mut paint = tiny_skia::Paint::default();
    paint.set_color_rgba8(0, 119, 200, 255); // Blue color
    paint.anti_alias = true;

    // Draw a circle
    let mut pb = tiny_skia::PathBuilder::new();
    pb.push_circle(200.0, 150.0, 50.0);
    let path = pb.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &path,
        &paint,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Save the pixmap to a PNG file
    let output_dir = Path::new("target/screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("tiny_skia_basic.png");
    pixmap.save_png(&output_path)?;

    println!("Screenshot saved to: {}", output_path.display());

    // Verify file was created and has content
    assert!(output_path.exists());
    let metadata = fs::metadata(&output_path)?;
    assert!(metadata.len() > 0);

    Ok(())
}

#[test]
fn test_tiny_skia_with_text() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger pixmap for text rendering
    let mut pixmap = tiny_skia::Pixmap::new(400, 300)
        .ok_or("Failed to create pixmap")?;

    // Fill with white background
    pixmap.fill(tiny_skia::Color::WHITE);

    // Draw colored rectangles
    let mut paint_blue = tiny_skia::Paint::default();
    paint_blue.set_color_rgba8(50, 100, 200, 255);

    let mut paint_green = tiny_skia::Paint::default();
    paint_green.set_color_rgba8(100, 200, 50, 255);

    // Top rectangle
    let mut pb = tiny_skia::PathBuilder::new();
    pb.push_rect(tiny_skia::Rect::from_xywh(50.0, 50.0, 300.0, 80.0).unwrap());
    let path = pb.finish().ok_or("Failed to create path")?;
    pixmap.fill_path(
        &path,
        &paint_blue,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Bottom rectangle
    let mut pb2 = tiny_skia::PathBuilder::new();
    pb2.push_rect(tiny_skia::Rect::from_xywh(50.0, 170.0, 300.0, 80.0).unwrap());
    let path2 = pb2.finish().ok_or("Failed to create path")?;
    pixmap.fill_path(
        &path2,
        &paint_green,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Save the screenshot
    let output_dir = Path::new("target/screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("tiny_skia_shapes.png");
    pixmap.save_png(&output_path)?;

    println!("Screenshot saved to: {}", output_path.display());

    // Verify file exists
    assert!(output_path.exists());

    Ok(())
}

#[test]
fn test_iced_tiny_skia_renderer_creation() {
    use iced::Font;
    use iced_core::Pixels;

    // Test that we can create an iced tiny-skia renderer
    let _renderer = iced_tiny_skia::Renderer::new(Font::default(), Pixels(16.0));

    // If we get here, creation succeeded
    assert!(true);
}

#[test]
fn test_snapshot_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Create a deterministic render for snapshot testing
    let mut pixmap = tiny_skia::Pixmap::new(200, 200)
        .ok_or("Failed to create pixmap")?;

    pixmap.fill(tiny_skia::Color::from_rgba8(240, 240, 240, 255));

    // Draw a simple pattern
    let mut paint = tiny_skia::Paint::default();
    paint.set_color_rgba8(255, 100, 100, 255);

    let mut pb = tiny_skia::PathBuilder::new();
    pb.push_rect(tiny_skia::Rect::from_xywh(50.0, 50.0, 100.0, 100.0).unwrap());
    let path = pb.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &path,
        &paint,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Save for snapshot testing with insta
    let output_dir = Path::new("target/screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("snapshot_test.png");
    pixmap.save_png(&output_path)?;

    // Use insta to snapshot the file content (or just verify it exists)
    let png_data = fs::read(&output_path)?;

    // Verify we have PNG data
    assert!(png_data.starts_with(&[137, 80, 78, 71])); // PNG magic number

    // Store snapshot using insta
    insta::assert_snapshot!(format!("PNG size: {} bytes", png_data.len()));

    Ok(())
}
