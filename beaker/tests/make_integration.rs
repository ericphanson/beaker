// tests/make_integration.rs
use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread::sleep,
};

fn fail(msg: impl AsRef<str>) -> ! {
    panic!("❌ FAIL: {}", msg.as_ref());
}
fn pass(msg: impl AsRef<str>) {
    eprintln!("✅ PASS: {}", msg.as_ref());
}

fn run_cmd<I, S>(
    program: &str,
    args: I,
    cwd: &Path,
    extra_env: &[(&str, &str)],
    quiet: bool,
) -> io::Result<(i32, String)>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);

    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    // path to beaker binary
    cmd.env("BEAKER", beaker_bin_path());

    if quiet {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    }

    let output = cmd.output()?;
    let code = output.status.code().unwrap_or(-1);
    let mut text = String::new();
    if quiet {
        text.push_str(&String::from_utf8_lossy(&output.stdout));
        if !output.stderr.is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&String::from_utf8_lossy(&output.stderr));
        }
    }
    Ok((code, text))
}

fn check_file_exists(path: &Path, desc: &str) {
    match fs::metadata(path) {
        Ok(md) if md.is_file() && md.len() > 0 => {
            pass(format!("{desc}: {} exists", path.display()))
        }
        _ => fail(format!("{desc}: {} missing or empty", path.display())),
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn check_up_to_date(target: &str, test_dir: &Path, stamp_dir: &Path) {
    let (_, out) = run_cmd(
        "make",
        [target],
        test_dir,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    )
    .unwrap_or_else(|e| fail(format!("make {target} failed to run: {e}")));
    if contains_any(&out, &["is up to date", "Nothing to be done"]) {
        pass(format!("No rebuild: {target}"));
    } else {
        fail(format!("Unexpected rebuild: {target} (output: {out})"));
    }
}

fn check_rebuilds(target: &str, test_dir: &Path, stamp_dir: &Path) {
    let (_, out) = run_cmd(
        "make",
        [target],
        test_dir,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    )
    .unwrap_or_else(|e| fail(format!("make {target} failed to run: {e}")));
    if contains_any(&out, &["Building", "Compiling"]) {
        pass(format!("Rebuilds correctly: {target}"));
    } else {
        fail(format!("Should have rebuilt: {target} (output: {out})"));
    }
}

fn read_to_string(path: &Path) -> String {
    let mut s = String::new();
    File::open(path)
        .and_then(|mut f| f.read_to_string(&mut s))
        .unwrap_or_else(|e| fail(format!("Failed reading {}: {e}", path.display())));
    s
}

fn depfile_has_format(path: &Path, needle_dep: &str) -> bool {
    // Validate: "<target>: ... example.jpg ..."
    for line in read_to_string(path).lines() {
        if let Some((_, rhs)) = line.split_once(':') {
            if rhs.contains(needle_dep) {
                return true;
            }
        }
    }
    false
}

fn copy_if_missing(src: &Path, dst: &Path) {
    if dst.exists() {
        return;
    }
    if src.exists() {
        fs::copy(src, dst).unwrap_or_else(|e| {
            fail(format!(
                "Failed to copy {} -> {}: {e}",
                src.display(),
                dst.display()
            ))
        });
    } else {
        fail(format!(
            "Missing fixture image: {} (and {} not present)",
            src.display(),
            dst.display()
        ));
    }
}

fn rm_rf(path: &Path) {
    let _ = fs::remove_file(path);
    if let Ok(md) = fs::metadata(path) {
        if md.is_dir() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn find_files_with_ext(dir: &Path, ext: &str, max: Option<usize>) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    fn walk(root: &Path, ext: &str, out: &mut Vec<PathBuf>, max: Option<usize>) -> io::Result<()> {
        if let Ok(rd) = fs::read_dir(root) {
            for entry in rd {
                let entry = entry?;
                let p = entry.path();
                if let Ok(ft) = entry.file_type() {
                    if ft.is_dir() {
                        walk(&p, ext, out, max)?;
                        if let Some(m) = max {
                            if out.len() >= m {
                                return Ok(());
                            }
                        }
                    } else if ft.is_file() && p.extension().map(|e| e == ext).unwrap_or(false) {
                        out.push(p);
                        if let Some(m) = max {
                            if out.len() >= m {
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    walk(dir, ext, &mut out, max)?;
    Ok(out)
}

fn count_stamp_files(dir: &Path) -> usize {
    find_files_with_ext(dir, "stamp", None)
        .unwrap_or_default()
        .len()
}
fn first_stamp_file(dir: &Path) -> Option<PathBuf> {
    find_files_with_ext(dir, "stamp", Some(1))
        .ok()
        .and_then(|v| v.into_iter().next())
}

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().to_path_buf()
}

// Where the Makefile for this test lives.
// Put your Makefile and .PHONY targets here (mirroring the Bash script).
fn test_dir() -> PathBuf {
    repo_root()
        .join("beaker")
        .join("tests")
        .join("make_integration")
}

fn beaker_bin_path() -> PathBuf {
    let beaker_path = env!("CARGO_BIN_EXE_beaker");

    PathBuf::from(beaker_path)
}

#[test]
fn make_integration_end_to_end() {
    // Log which binary we are using with how long ago the binary was built
    let built_time_ago = {
        let metadata = fs::metadata(beaker_bin_path()).unwrap();
        let modified = metadata.modified().unwrap();
        let now = std::time::SystemTime::now();
        now.duration_since(modified).unwrap()
    };
    let built_time_ago_secs = built_time_ago.as_secs_f64();
    eprintln!(
        "Using beaker binary: {}, built {:.1}s ago",
        beaker_bin_path().display(),
        built_time_ago_secs
    );

    // Run everything as a single test to avoid parallelism races.
    let td = test_dir();
    if !td.exists() {
        fail(format!(
            "Expected test directory with Makefile at: {}",
            td.display()
        ));
    }
    // Prepare fixtures
    let repo = repo_root();
    copy_if_missing(&repo.join("example.jpg"), &td.join("example.jpg"));
    copy_if_missing(
        &repo.join("example-2-birds.jpg"),
        &td.join("example-2-birds.jpg"),
    );

    // Isolated stamp dir
    let stamp_dir = td.join("test_stamps");
    env::set_var("BEAKER_STAMP_DIR", &stamp_dir);

    eprintln!("=== Make Integration Test ===");

    // Clean start
    let _ = run_cmd("make", ["clean"], &td, &[], true);
    rm_rf(&stamp_dir);

    eprintln!("\n=== Core Dependency Tracking Tests ===");

    // Test 1: Initial build creates expected files
    eprintln!("Test 1: Initial build");
    run_cmd(
        "make",
        ["all"],
        &td,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    )
    .unwrap_or_else(|e| fail(format!("make all failed: {e}")));
    check_file_exists(&td.join("example_cutout.png"), "Cutout output");
    check_file_exists(&td.join("example_bounding-box.jpg"), "Detection output");
    check_file_exists(&td.join("example_cutout.png.d"), "Cutout depfile");
    check_file_exists(&td.join("example_bounding-box.jpg.d"), "Detection depfile");

    // Test 2: No spurious rebuilds
    eprintln!("\nTest 2: No spurious rebuilds");
    check_up_to_date("example_cutout.png", &td, &stamp_dir);
    check_up_to_date("example_bounding-box.jpg", &td, &stamp_dir);
    check_up_to_date("all", &td, &stamp_dir);

    // Test 3: Rebuild when input changes
    eprintln!("\nTest 3: Rebuild after input change");
    // touch example.jpg
    // sleep 100ms to ensure timestamp changes
    sleep(std::time::Duration::from_millis(100));
    _ = run_cmd("touch", ["example.jpg"], &td, &[], true);

    check_rebuilds("example_cutout.png", &td, &stamp_dir);

    // Test 4: Verify depfile content format
    eprintln!("\nTest 4: Depfile content validation");
    for depfile in [
        td.join("example_cutout.png.d"),
        td.join("example_bounding-box.jpg.d"),
    ] {
        if depfile_has_format(&depfile, "example.jpg") {
            pass(format!("Depfile format: {}", depfile.display()));
        } else {
            let content = read_to_string(&depfile);
            fail(format!(
                "Invalid depfile format: {} (content: {})",
                depfile.display(),
                content
            ));
        }
    }

    // Test 5: Stamp files created
    eprintln!("\nTest 5: Stamp file generation");
    let n = count_stamp_files(&stamp_dir);
    if n == 0 {
        fail(format!("No stamp files found in {}", stamp_dir.display()));
    }
    pass(format!("Stamp files created ({n} files)"));

    // Test 6: Cross-tool dependency isolation
    eprintln!("\nTest 6: Cross-tool dependency isolation");
    let _ = run_cmd("make", ["clean"], &td, &[], true);
    let _ = run_cmd(
        "make",
        ["example_bounding-box.jpg"],
        &td,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    );
    let _ = run_cmd(
        "make",
        ["example_cutout.png"],
        &td,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    );
    check_up_to_date("example_bounding-box.jpg", &td, &stamp_dir);

    // Test 7: Verify specific targets work independently
    eprintln!("\nTest 7: Individual target builds");
    let _ = run_cmd("make", ["clean"], &td, &[], true);
    rm_rf(&stamp_dir); // Clean stamp directory to avoid stale dependencies
    let _ = run_cmd(
        "make",
        ["example_crop_head.jpg"],
        &td,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    );
    check_file_exists(&td.join("example_crop_head.jpg"), "Crop target output");
    check_file_exists(&td.join("example_crop_head.jpg.d"), "Crop target depfile");
    check_up_to_date("example_crop_head.jpg", &td, &stamp_dir);

    eprintln!("\n=== Advanced Dependency Tests ===");

    // Test 8: Stamp file changes trigger rebuilds
    eprintln!("Test 8: Stamp file dependency tracking");
    let _ = run_cmd("make", ["clean"], &td, &[], true);
    let _ = run_cmd(
        "make",
        ["example_cutout.png"],
        &td,
        &[("BEAKER_STAMP_DIR", &stamp_dir.to_string_lossy())],
        true,
    );

    if let Some(stamp) = first_stamp_file(&stamp_dir) {
        // touch stamp after 100ms to ensure timestamp changes
        sleep(std::time::Duration::from_millis(100));
        _ = run_cmd("touch", [stamp], &td, &[], true);
        check_rebuilds("example_cutout.png", &td, &stamp_dir);
        pass("Stamp file dependency tracking");
    } else {
        fail("No stamp file found for dependency test");
    }

    // Cleanup and summary
    eprintln!("\n=== Cleanup and Summary ===");
    let _ = run_cmd("make", ["clean"], &td, &[], true);
    rm_rf(&stamp_dir);
    let _ = fs::remove_file(td.join("example.jpg"));
    let _ = fs::remove_file(td.join("example-2-birds.jpg"));

    eprintln!("✅ All tests passed!");
}
