// Screenshot tests that create mockups of the hello-world app UI
// These demonstrate headless rendering of app-like interfaces
use std::fs;
use std::path::Path;

/// Creates a mockup screenshot of the hello world counter app at a specific state
fn create_counter_screenshot(
    counter_value: i32,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a pixmap matching typical app size
    let width = 400;
    let height = 300;
    let mut pixmap = tiny_skia::Pixmap::new(width, height)
        .ok_or("Failed to create pixmap")?;

    // Fill with light gray background (typical window background)
    pixmap.fill(tiny_skia::Color::from_rgba8(240, 240, 240, 255));

    // Draw a centered container area (white)
    let container_rect = tiny_skia::Rect::from_xywh(50.0, 50.0, 300.0, 200.0)
        .ok_or("Invalid rect")?;

    let mut paint_white = tiny_skia::Paint::default();
    paint_white.set_color(tiny_skia::Color::WHITE);

    let mut pb = tiny_skia::PathBuilder::new();
    pb.push_rect(container_rect);
    let path = pb.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &path,
        &paint_white,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Draw title area (represents "Hello, World!" text area)
    let title_rect = tiny_skia::Rect::from_xywh(100.0, 80.0, 200.0, 50.0)
        .ok_or("Invalid rect")?;

    let mut paint_title = tiny_skia::Paint::default();
    paint_title.set_color_rgba8(70, 130, 180, 255); // Steel blue
    paint_title.anti_alias = true;

    let mut pb_title = tiny_skia::PathBuilder::new();
    pb_title.push_rect(title_rect);
    let title_path = pb_title.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &title_path,
        &paint_title,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Draw counter display area
    let counter_rect = tiny_skia::Rect::from_xywh(125.0, 150.0, 150.0, 30.0)
        .ok_or("Invalid rect")?;

    let mut paint_counter = tiny_skia::Paint::default();
    paint_counter.set_color_rgba8(100, 100, 100, 255); // Dark gray

    let mut pb_counter = tiny_skia::PathBuilder::new();
    pb_counter.push_rect(counter_rect);
    let counter_path = pb_counter.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &counter_path,
        &paint_counter,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Draw button areas
    // Increment button (green)
    let inc_button = tiny_skia::Rect::from_xywh(80.0, 200.0, 110.0, 35.0)
        .ok_or("Invalid rect")?;

    let mut paint_inc = tiny_skia::Paint::default();
    paint_inc.set_color_rgba8(76, 175, 80, 255); // Green
    paint_inc.anti_alias = true;

    let mut pb_inc = tiny_skia::PathBuilder::new();
    pb_inc.push_rect(inc_button);
    let inc_path = pb_inc.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &inc_path,
        &paint_inc,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Decrement button (red)
    let dec_button = tiny_skia::Rect::from_xywh(210.0, 200.0, 110.0, 35.0)
        .ok_or("Invalid rect")?;

    let mut paint_dec = tiny_skia::Paint::default();
    paint_dec.set_color_rgba8(244, 67, 54, 255); // Red
    paint_dec.anti_alias = true;

    let mut pb_dec = tiny_skia::PathBuilder::new();
    pb_dec.push_rect(dec_button);
    let dec_path = pb_dec.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &dec_path,
        &paint_dec,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Add value indicator (small colored bar representing the counter value)
    let value_indicator_width = (counter_value.abs().min(10) as f32 * 10.0).max(5.0);
    let indicator_color = if counter_value >= 0 {
        tiny_skia::Color::from_rgba8(76, 175, 80, 255) // Green for positive
    } else {
        tiny_skia::Color::from_rgba8(244, 67, 54, 255) // Red for negative
    };

    let indicator_rect = tiny_skia::Rect::from_xywh(
        200.0 - value_indicator_width / 2.0,
        190.0,
        value_indicator_width,
        5.0,
    ).ok_or("Invalid rect")?;

    let mut paint_indicator = tiny_skia::Paint::default();
    paint_indicator.set_color(indicator_color);

    let mut pb_indicator = tiny_skia::PathBuilder::new();
    pb_indicator.push_rect(indicator_rect);
    let indicator_path = pb_indicator.finish().ok_or("Failed to create path")?;

    pixmap.fill_path(
        &indicator_path,
        &paint_indicator,
        tiny_skia::FillRule::Winding,
        tiny_skia::Transform::identity(),
        None,
    );

    // Save the screenshot
    pixmap.save_png(output_path)?;

    Ok(())
}

#[test]
fn test_counter_initial_state() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = Path::new("screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("counter_initial.png");
    create_counter_screenshot(0, &output_path)?;

    println!("Screenshot saved to: {}", output_path.display());
    assert!(output_path.exists());

    Ok(())
}

#[test]
fn test_counter_positive_value() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = Path::new("screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("counter_positive.png");
    create_counter_screenshot(5, &output_path)?;

    println!("Screenshot saved to: {}", output_path.display());
    assert!(output_path.exists());

    Ok(())
}

#[test]
fn test_counter_negative_value() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = Path::new("screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("counter_negative.png");
    create_counter_screenshot(-3, &output_path)?;

    println!("Screenshot saved to: {}", output_path.display());
    assert!(output_path.exists());

    Ok(())
}

#[test]
fn test_counter_high_value() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = Path::new("screenshots");
    fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join("counter_high_value.png");
    create_counter_screenshot(10, &output_path)?;

    println!("Screenshot saved to: {}", output_path.display());
    assert!(output_path.exists());

    Ok(())
}

