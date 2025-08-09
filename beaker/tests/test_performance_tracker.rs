use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Test performance tracking
#[derive(Debug)]
pub struct TestPerformanceTracker {
    detect_invocations: u32,
    cutout_invocations: u32,
    total_test_time: Duration,
    slowest_tests: Vec<(String, Duration)>,
    completed_tests: HashMap<String, bool>,
    expected_tests: Vec<String>,
    wall_clock_start: Option<Instant>,
    wall_clock_end: Option<Instant>,
}

impl TestPerformanceTracker {
    pub fn new(test_names: Vec<String>) -> Self {
        let mut completed_tests = HashMap::new();
        // Initialize all tests as not completed
        for test_name in &test_names {
            completed_tests.insert(test_name.clone(), false);
        }

        Self {
            detect_invocations: 0,
            cutout_invocations: 0,
            total_test_time: Duration::ZERO,
            slowest_tests: Vec::new(),
            completed_tests,
            expected_tests: test_names,
            wall_clock_start: None,
            wall_clock_end: None,
        }
    }

    pub fn record_test(&mut self, test_name: &str, tool: &str, duration: Duration) -> bool {
        // Start wall clock on first test
        if self.wall_clock_start.is_none() {
            self.wall_clock_start = Some(Instant::now());
        }

        // Track per-tool invocations
        match tool {
            "detect" => self.detect_invocations += 1,
            "cutout" => self.cutout_invocations += 1,
            "both" => {
                self.detect_invocations += 1;
                self.cutout_invocations += 1;
            }
            _ => {}
        }

        // Add to total time
        self.total_test_time += duration;

        // Track slowest tests (keep top 5)
        self.slowest_tests.push((test_name.to_string(), duration));
        self.slowest_tests.sort_by(|a, b| b.1.cmp(&a.1));
        if self.slowest_tests.len() > 5 {
            self.slowest_tests.truncate(5);
        }

        // Mark test as completed
        self.completed_tests.insert(test_name.to_string(), true);

        // Check if all tests are completed
        let all_done = self.all_tests_completed();
        if all_done {
            self.wall_clock_end = Some(Instant::now());
        }
        all_done
    }

    fn all_tests_completed(&self) -> bool {
        self.expected_tests.iter().all(|test_name| {
            self.completed_tests
                .get(test_name)
                .copied()
                .unwrap_or(false)
        })
    }

    pub fn print_summary(&self) {
        let wall_clock_duration =
            if let (Some(start), Some(end)) = (self.wall_clock_start, self.wall_clock_end) {
                end.duration_since(start)
            } else {
                Duration::ZERO
            };

        println!("\n📊 Test Performance Summary:");
        println!(
            "  Wall clock time: {:.2}s (actual time elapsed)",
            wall_clock_duration.as_secs_f64()
        );
        println!(
            "  Total CPU time: {:.2}s (cumulative across all tests)",
            self.total_test_time.as_secs_f64()
        );
        println!("  Detect model invocations: {}", self.detect_invocations);
        println!("  Cutout model invocations: {}", self.cutout_invocations);

        if !self.slowest_tests.is_empty() {
            println!("  Slowest tests:");
            for (name, duration) in &self.slowest_tests {
                println!("    - {}: {:.2}s", name, duration.as_secs_f64());
            }
        }

        // Performance warnings - focus on wall clock time for development feedback
        if wall_clock_duration.as_secs() > 60 {
            println!(
                "  ⚠️  Wall clock time exceeded 60s target ({:.2}s)",
                wall_clock_duration.as_secs_f64()
            );
        }
        if self.cutout_invocations > 10 {
            println!(
                "  ⚠️  High cutout model usage ({} invocations) - consider optimization",
                self.cutout_invocations
            );
        }

        println!("✅ All metadata validation tests completed!");
    }
}

/// Global performance tracker instance
pub static GLOBAL_PERFORMANCE_TRACKER: Mutex<Option<TestPerformanceTracker>> = Mutex::new(None);

/// Initialize the global performance tracker with expected test names
pub fn initialize_performance_tracker(test_names: Vec<String>) {
    if let Ok(mut tracker_option) = GLOBAL_PERFORMANCE_TRACKER.lock() {
        *tracker_option = Some(TestPerformanceTracker::new(test_names));
    }
}

/// Record test performance in the global tracker
pub fn record_test_performance(test_name: &str, tool: &str, duration: Duration) {
    // Warn if any single test takes > 30 seconds (more reasonable threshold)
    if duration.as_secs() > 30 {
        eprintln!(
            "⚠️  Slow test: {} took {:.2}s",
            test_name,
            duration.as_secs_f64()
        );
    }

    // Record in global tracker
    if let Ok(mut tracker_option) = GLOBAL_PERFORMANCE_TRACKER.lock() {
        if let Some(tracker) = tracker_option.as_mut() {
            let is_last_test = tracker.record_test(test_name, tool, duration);

            // Print summary if this was the last test
            if is_last_test {
                tracker.print_summary();
            }
        }
    }
}
