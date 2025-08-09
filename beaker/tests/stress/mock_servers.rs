//! Mock HTTP/TCP servers with deterministic failure injection

use httpmock::{Mock, MockServer, Then, When};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::stress::fixtures::TestFixtures;

/// Types of failures that can be injected
#[derive(Clone, Debug)]
pub enum FailureEvent {
    Success,
    HttpError(u16),           // HTTP status codes via httpmock
    CorruptedChecksum,        // Wrong SHA-256 via httpmock
    TcpConnectionRefused,     // Simulate connection refused
    TcpMidStreamAbort(usize), // Simulate partial download
    TcpHeaderThenClose,       // Simulate headers only
}

/// Controller for deterministic failure injection
pub struct FailureController {
    failure_sequence: Vec<FailureEvent>,
    current_request: AtomicUsize,
}

impl FailureController {
    pub fn new(pattern: Vec<FailureEvent>) -> Self {
        Self {
            failure_sequence: pattern,
            current_request: AtomicUsize::new(0),
        }
    }

    pub fn next_response(&self) -> FailureEvent {
        let index = self.current_request.fetch_add(1, Ordering::SeqCst);
        self.failure_sequence[index % self.failure_sequence.len()].clone()
    }
}

/// Mock server manager that handles HTTP level failures
pub struct MockServerManager {
    http_server: MockServer,
    fixtures: TestFixtures,
}

impl MockServerManager {
    pub fn new(failure_pattern: Vec<FailureEvent>) -> Self {
        let fixtures = TestFixtures::new();
        let http_server = MockServer::start();

        let mut manager = Self {
            http_server,
            fixtures,
        };

        manager.configure_mocks(failure_pattern);
        manager
    }

    pub fn http_base_url(&self) -> String {
        self.http_server.base_url()
    }

    fn configure_mocks(&mut self, failure_pattern: Vec<FailureEvent>) {
        let failure_controller = Arc::new(FailureController::new(failure_pattern));
        let fixtures = self.fixtures.clone();

        let failure_controller_clone = Arc::clone(&failure_controller);

        // Simple mock configuration for Phase 1
        // Note: This is a simplified version. Full implementation with dynamic responses
        // will be completed in Phase 2
        let _mock = self.http_server.mock(|when, then| {
            when.path("/small-test-model.onnx");
            match failure_controller_clone.next_response() {
                FailureEvent::Success => {
                    then.status(200).body(fixtures.small_model_bytes);
                }
                FailureEvent::HttpError(code) => {
                    then.status(code);
                }
                FailureEvent::CorruptedChecksum => {
                    then.status(200).body(&fixtures.corrupted_model_bytes);
                }
                // For TCP-level failures, just return HTTP errors for now
                FailureEvent::TcpConnectionRefused => {
                    then.status(503);
                }
                FailureEvent::TcpMidStreamAbort(_) => {
                    then.status(502);
                }
                FailureEvent::TcpHeaderThenClose => {
                    then.status(500);
                }
            }
        });
    }
}

// Clone implementation for TestFixtures
impl Clone for TestFixtures {
    fn clone(&self) -> Self {
        Self {
            small_model_bytes: self.small_model_bytes,
            small_model_sha256: self.small_model_sha256,
            corrupted_model_bytes: self.corrupted_model_bytes.clone(),
            large_model_bytes: self.large_model_bytes,
            large_model_sha256: self.large_model_sha256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_controller() {
        let pattern = vec![
            FailureEvent::Success,
            FailureEvent::HttpError(500),
            FailureEvent::CorruptedChecksum,
        ];

        let controller = FailureController::new(pattern);

        // Test cycling through pattern
        assert!(matches!(controller.next_response(), FailureEvent::Success));
        assert!(matches!(
            controller.next_response(),
            FailureEvent::HttpError(500)
        ));
        assert!(matches!(
            controller.next_response(),
            FailureEvent::CorruptedChecksum
        ));
        assert!(matches!(controller.next_response(), FailureEvent::Success)); // Cycles back
    }

    #[test]
    fn test_mock_server_creation() {
        let pattern = vec![FailureEvent::Success, FailureEvent::HttpError(404)];

        let _mock_manager = MockServerManager::new(pattern);
        // Test passes if no panic occurs
    }
}
