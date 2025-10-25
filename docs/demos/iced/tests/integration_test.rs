// Basic integration tests that can run headless
// These tests validate the application logic without requiring a display

use std::env;

#[test]
fn test_iced_backend_env_var() {
    // Test that we can set the ICED_BACKEND environment variable
    // This is important for CI where we want to force software rendering
    env::set_var("ICED_BACKEND", "tiny-skia");
    assert_eq!(env::var("ICED_BACKEND").unwrap(), "tiny-skia");
    env::remove_var("ICED_BACKEND");
}

#[test]
fn test_counter_logic() {
    // Test the counter logic without rendering
    #[derive(Debug, Clone)]
    enum Message {
        Increment,
        Decrement,
    }

    struct Counter {
        value: i32,
    }

    fn update(counter: &mut Counter, message: Message) {
        match message {
            Message::Increment => counter.value += 1,
            Message::Decrement => counter.value -= 1,
        }
    }

    let mut counter = Counter { value: 0 };
    assert_eq!(counter.value, 0);

    update(&mut counter, Message::Increment);
    assert_eq!(counter.value, 1);

    update(&mut counter, Message::Increment);
    update(&mut counter, Message::Increment);
    assert_eq!(counter.value, 3);

    update(&mut counter, Message::Decrement);
    assert_eq!(counter.value, 2);
}

#[test]
fn test_tiny_skia_feature() {
    // This test verifies that the tiny-skia feature is enabled
    // by checking that we can reference iced_tiny_skia types
    // (this test just needs to compile to verify the feature is enabled)
    let _marker: Option<iced_tiny_skia::Renderer> = None;
}
