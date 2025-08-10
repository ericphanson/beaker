// beaker-stamp/src/lib.rs

pub trait Stamp {
    /// Canonical JSON of *only* the #[stamp]-marked fields.
    fn stamp_value(&self) -> serde_json::Value;

    /// "sha256:<hex>" over canonical JSON bytes.
    fn stamp_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let v = self.stamp_value();
        let s = canonical_json::to_string(&v);
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        format!("sha256:{:x}", h.finalize())
    }
}

/// Minimal canonical JSON (sorted object keys; stable numbers)
pub mod canonical_json {
    use serde_json::{Map, Number, Value};
    pub fn to_string(v: &Value) -> String {
        serde_json::to_string(&normalize(v)).expect("canonical json")
    }
    fn normalize(v: &Value) -> Value {
        match v {
            Value::Object(map) => {
                let mut out = Map::new();
                let mut keys: Vec<_> = map.keys().cloned().collect();
                keys.sort();
                for k in keys {
                    out.insert(k.clone(), normalize(&map[&k]));
                }
                Value::Object(out)
            }
            Value::Array(a) => Value::Array(a.iter().map(normalize).collect()),
            Value::Number(n) => stable_number(n),
            _ => v.clone(),
        }
    }
    fn stable_number(n: &Number) -> Value {
        // serde_json already prints a stable form; if you need rounding,
        // do it via #[stamp(with = "...")] on the field.
        Value::Number(n.clone())
    }
}

pub mod paths {
    use std::path::PathBuf;
    /// OS-appropriate cache dir: .../beaker/stamps
    /// Can be overridden with BEAKER_STAMP_DIR environment variable
    pub fn stamp_dir() -> PathBuf {
        if let Ok(custom_dir) = std::env::var("BEAKER_STAMP_DIR") {
            let p = PathBuf::from(custom_dir);
            std::fs::create_dir_all(&p).ok();
            return p;
        }

        let base = directories::BaseDirs::new().expect("dirs");
        let mut p = base.cache_dir().to_path_buf();
        p.push("beaker");
        p.push("stamps");
        std::fs::create_dir_all(&p).ok();
        p
    }
}

pub mod write {
    use std::{fs, io::Write, path::Path};

    /// Atomically write bytes to `path` iff content differs (preserve mtime otherwise).
    pub fn write_atomic_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        if let Ok(prev) = fs::read(path) {
            if prev == bytes {
                return Ok(());
            }
        }
        let tmp = path.with_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Helper to write a one-line config stamp into the hidden cache dir.
pub fn write_cfg_stamp(subcmd: &str, cfg: &impl Stamp) -> std::io::Result<std::path::PathBuf> {
    use std::path::PathBuf;
    let hash = cfg.stamp_hash(); // "sha256:…"
    let fname = format!(
        "cfg-{}-{}.stamp",
        subcmd,
        hash.trim_start_matches("sha256:")
    );
    let path: PathBuf = paths::stamp_dir().join(fname);
    let content = format!("cfg={subcmd} {hash}\n").into_bytes();
    write::write_atomic_if_changed(&path, &content)?;
    Ok(path)
}

#[cfg(test)]
mod test;
