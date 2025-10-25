use iced::widget::{button, column, container, text};
use iced::{Alignment, Element, Length, Task};

pub fn main() -> iced::Result {
    iced::application("Hello World - Iced Demo", update, view)
        .centered()
        .run()
}

#[derive(Debug, Clone)]
enum Message {
    Increment,
    Decrement,
}

struct Counter {
    value: i32,
}

impl Default for Counter {
    fn default() -> Self {
        Self { value: 0 }
    }
}

fn update(counter: &mut Counter, message: Message) -> Task<Message> {
    match message {
        Message::Increment => counter.value += 1,
        Message::Decrement => counter.value -= 1,
    }
    Task::none()
}

fn view(counter: &Counter) -> Element<'_, Message> {
    container(
        column![
            text("Hello, World!").size(50),
            text(format!("Counter: {}", counter.value)).size(30),
            button("Increment").on_press(Message::Increment),
            button("Decrement").on_press(Message::Decrement),
        ]
        .spacing(20)
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .center_x(Length::Fill)
    .center_y(Length::Fill)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increment() {
        let mut counter = Counter::default();
        assert_eq!(counter.value, 0);

        let _ = update(&mut counter, Message::Increment);
        assert_eq!(counter.value, 1);

        let _ = update(&mut counter, Message::Increment);
        assert_eq!(counter.value, 2);
    }

    #[test]
    fn test_counter_decrement() {
        let mut counter = Counter::default();
        assert_eq!(counter.value, 0);

        let _ = update(&mut counter, Message::Decrement);
        assert_eq!(counter.value, -1);
    }
}
