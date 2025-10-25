#!/usr/bin/env python3
"""
Generate example snapshot images to demonstrate what egui_kittest snapshots would look like.

These are simplified representations. Real snapshots generated on systems with GPU
access will show the actual rendered egui widgets.
"""

from PIL import Image, ImageDraw, ImageFont

def create_snapshot(filename, title, widgets_desc):
    """Create a simple demonstration snapshot image"""
    # Create image with light gray background (similar to egui default)
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        normal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()

    # Draw title
    draw.text((10, 10), title, fill=(0, 0, 0), font=title_font)

    # Draw widget descriptions
    y = 50
    for desc in widgets_desc:
        draw.text((10, y), desc, fill=(50, 50, 50), font=normal_font)
        y += 25

    # Add a subtle note at the bottom
    note_text = "Example snapshot (real snapshots require GPU)"
    draw.text((10, height - 25), note_text, fill=(150, 150, 150), font=normal_font)

    # Save the image
    img.save(filename)
    print(f"Created {filename}")

# Generate example snapshots
snapshots_dir = "/home/user/beaker/docs/demos/egui/tests/snapshots"

create_snapshot(
    f"{snapshots_dir}/hello_world_initial.png",
    "Hello, World!",
    ["Your name: [___________]"]
)

create_snapshot(
    f"{snapshots_dir}/simple_heading.png",
    "Hello, World!",
    []
)

create_snapshot(
    f"{snapshots_dir}/text_input_widget.png",
    "",
    ["Your name: [___________]"]
)

create_snapshot(
    f"{snapshots_dir}/button_before_click.png",
    "",
    ["[ Click me ]"]
)

create_snapshot(
    f"{snapshots_dir}/button_after_click.png",
    "",
    ["[ Click me ] (clicked)"]
)

create_snapshot(
    f"{snapshots_dir}/multiple_widgets.png",
    "Demo Application",
    [
        "─────────────────────",
        "This is a label",
        "[ Button 1 ]  [ Button 2 ]",
        "☐ Checkbox"
    ]
)

create_snapshot(
    f"{snapshots_dir}/colored_text.png",
    "Styled Text Demo",
    [
        "Red text (in red)",
        "Green text (in green)",
        "Blue text (in blue)"
    ]
)

print("\n✓ Example snapshots created successfully!")
print("Note: These are simplified examples. Real snapshots generated with")
print("UPDATE_SNAPSHOTS=true on a GPU-enabled system will show actual rendered widgets.")
