//! Images are decoded with their EXIF Orientation tag applied, so every command sees the
//! pixels upright, as an image viewer shows them.

use beaker::image_input::load_image;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn encode_jpeg(img: &image::RgbImage) -> Vec<u8> {
    let mut jpeg = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut jpeg),
        image::ImageFormat::Jpeg,
    )
    .unwrap();
    jpeg
}

/// Add an EXIF Orientation tag to a JPEG that has no EXIF
fn with_exif_orientation(mut jpeg: Vec<u8>, orientation: u8) -> Vec<u8> {
    // APP1 segment: a little-endian TIFF header and one IFD holding the Orientation tag
    let mut app1 = vec![0xFF, 0xE1, 0x00, 0x22];
    app1.extend(b"Exif\0\0II*\0");
    app1.extend(8u32.to_le_bytes()); // offset of the IFD
    app1.extend(1u16.to_le_bytes()); // one entry
    app1.extend(0x0112u16.to_le_bytes()); // tag: Orientation
    app1.extend(3u16.to_le_bytes()); // type: SHORT
    app1.extend(1u32.to_le_bytes()); // count
    app1.extend(u32::from(orientation).to_le_bytes()); // value
    app1.extend(0u32.to_le_bytes()); // no next IFD
    jpeg.splice(2..2, app1); // directly after the SOI marker
    jpeg
}

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const WHITE: [u8; 3] = [255, 255, 255];

/// Load a 48x32 JPEG whose quadrants are, as stored: red (top-left), green (top-right),
/// blue (bottom-left), white (bottom-right). Returns the loaded dimensions and the colours at
/// the centres of the four quadrants, in the same order.
fn load_quadrants(orientation: Option<u8>) -> ((u32, u32), [[u8; 3]; 4]) {
    let stored = image::RgbImage::from_fn(48, 32, |x, y| {
        image::Rgb(match (x < 24, y < 16) {
            (true, true) => RED,
            (false, true) => GREEN,
            (true, false) => BLUE,
            (false, false) => WHITE,
        })
    });
    let mut jpeg = encode_jpeg(&stored);
    if let Some(orientation) = orientation {
        jpeg = with_exif_orientation(jpeg, orientation);
    }
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("quadrants.jpg");
    std::fs::write(&path, jpeg).unwrap();

    let img = load_image(&path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    // JPEG is lossy, so snap each channel to 0 or 255
    let colour = |x, y| img.get_pixel(x, y).0.map(|c| if c > 127 { 255 } else { 0 });
    let quadrants = [
        colour(w / 4, h / 4),
        colour(3 * w / 4, h / 4),
        colour(w / 4, 3 * h / 4),
        colour(3 * w / 4, 3 * h / 4),
    ];
    ((w, h), quadrants)
}

#[test]
fn test_load_image_applies_exif_orientation() {
    // No EXIF, and Orientation 1: pixels as stored
    for orientation in [None, Some(1)] {
        assert_eq!(
            load_quadrants(orientation),
            ((48, 32), [RED, GREEN, BLUE, WHITE]),
            "orientation {orientation:?}"
        );
    }
    // Orientation 6: a viewer rotates the stored pixels 90 degrees clockwise
    assert_eq!(
        load_quadrants(Some(6)),
        ((32, 48), [BLUE, RED, WHITE, GREEN])
    );
    // Orientation 8: a viewer rotates the stored pixels 90 degrees counter-clockwise
    assert_eq!(
        load_quadrants(Some(8)),
        ((32, 48), [GREEN, WHITE, RED, BLUE])
    );
    // Orientation 3: a viewer rotates the stored pixels 180 degrees
    assert_eq!(
        load_quadrants(Some(3)),
        ((48, 32), [WHITE, BLUE, GREEN, RED])
    );
}

/// Run a beaker command on an image and return its metadata section
fn run_beaker(tool: &str, image: &Path, output_dir: &Path) -> toml::Table {
    let output = Command::new(env!("CARGO_BIN_EXE_beaker"))
        .args(["--metadata", "--output-dir"])
        .arg(output_dir)
        .args([tool, "--device", "cpu"])
        .arg(image)
        .output()
        .expect("Failed to execute beaker command");
    assert!(
        output.status.success(),
        "{tool} command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stem = image.file_stem().unwrap().to_str().unwrap();
    let toml_path = output_dir.join(format!("{stem}.beaker.toml"));
    let metadata: toml::Table =
        toml::from_str(&std::fs::read_to_string(&toml_path).unwrap()).unwrap();
    metadata[tool].as_table().unwrap().clone()
}

/// (x1, y1, x2, y2) of the head detection
fn head_box(detect: &toml::Table) -> [f64; 4] {
    let head = detect["detections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["class_name"].as_str() == Some("head"))
        .expect("No head detected");
    ["x1", "y1", "x2", "y2"].map(|key| head[key].as_float().unwrap())
}

/// A photo stored sideways with an Orientation tag must give the same results as the upright
/// photo, in upright coordinates
#[test]
fn test_commands_apply_exif_orientation() {
    let temp_dir = TempDir::new().unwrap();
    let upright_path = Path::new("../example.jpg");
    let upright = image::open(upright_path).unwrap();
    let (width, height) = (upright.width(), upright.height());
    assert_ne!(width, height);

    // Store the pixels rotated 90 degrees clockwise, tagged as needing the opposite rotation
    let sideways_path = temp_dir.path().join("sideways.jpg");
    let sideways = with_exif_orientation(encode_jpeg(&upright.rotate90().to_rgb8()), 8);
    std::fs::write(&sideways_path, sideways).unwrap();

    for tool in ["detect", "quality", "cutout"] {
        let metadata = run_beaker(tool, &sideways_path, temp_dir.path());
        let dimension = |key: &str| metadata[key].as_integer().unwrap();
        assert_eq!(
            (dimension("input_img_width"), dimension("input_img_height")),
            (width.into(), height.into()),
            "{tool} did not apply the orientation"
        );

        if tool == "detect" {
            let expected = head_box(&run_beaker(tool, upright_path, temp_dir.path()));
            let actual = head_box(&metadata);
            // The sideways copy is re-encoded, which moves the box by a few pixels
            let max_difference = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0, f64::max);
            assert!(
                max_difference < 0.02 * f64::from(width),
                "Head box {actual:?} of the sideways copy differs from {expected:?}"
            );
        }
    }

    let cutout = image::image_dimensions(temp_dir.path().join("sideways_cutout.png")).unwrap();
    assert_eq!(cutout, (width, height));
}
