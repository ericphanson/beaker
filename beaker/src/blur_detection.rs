use image::{ImageBuffer, Luma, RgbImage};
use imageproc::filter::{filter3x3, gaussian_blur_f32};
use ndarray::{Array2, Array4, Axis};
use serde::Serialize;

/// ------------------------- tunables -------------------------
const ALPHA: f32 = 0.7; // downweight strength for blur (W = 1 - ALPHA * P)
const MIN_WEIGHT: f32 = 0.2; // floor on per-cell weights
const TAU_TEN_224: f32 = 0.02; // threshold for 224x224 in [0,1]
const S_REF: f32 = 96.0; // size prior (min bbox side in native px to be "fully reliable")
const COV_REF: f32 = 4.0; // coverage prior baseline (~cells) → 2x2 cells
const ROI_SAMPLES: usize = 8; // bilinear samples per axis inside bbox for ROI pooling
const GAUSS_SIGMA_NATIVE: f32 = 1.0; // denoise native crop before Tenengrad/detail
const BETA: f32 = 1.2; // 1.1–1.6 typical; larger => harsher blur probability
const EPS_T: f32 = 1e-12; // safety epsilon for divisions
                          // optional: ensure we never return an exact 0 probability (avoids W==1.0)
const P_FLOOR: f32 = 0.0; // try 0.01..0.02 if you want a tiny universal penalty
/// ------------------------- basic types -------------------------

type GrayF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

#[derive(Copy, Clone, Debug)]
pub struct BBoxF {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionQuality {
    /// final per-detection quality (same scale as PaQ map, e.g., 0..100)
    pub quality: f32,
    /// components for debugging / thresholding
    pub q_roi_mean: f32, // mean PaQ map inside bbox (ROI-pooled)
    pub w_roi_mean: f32,     // mean blur weight inside bbox (ROI-pooled)
    pub detail_prob: f32,    // native-resolution detail gate (0..1)
    pub size_prior: f32,     // 0..1
    pub coverage_prior: f32, // 0..1
}

use image::Rgb;
/// ------------------------- utils -------------------------
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// ----- tiny utils -----
fn percentile(mut vals: Vec<f32>, p: f32) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((vals.len() as f32 - 1.0) * p).round() as usize;
    vals[k]
}

/// ----- 224 background from NCHW -----
pub fn nchw_to_rgb_224(x: &Array4<f32>) -> RgbImage {
    assert_eq!(x.shape(), &[1, 3, 224, 224], "expected [1,3,224,224]");
    let r = x.index_axis(Axis(1), 0);
    let g = x.index_axis(Axis(1), 1);
    let b = x.index_axis(Axis(1), 2);
    let mut img = RgbImage::new(224, 224);
    for i in 0..224 {
        for j in 0..224 {
            // assume in [0,1]; clamp just in case
            let rr = (clampf(r[[0, i, j]], 0.0, 1.0) * 255.0).round() as u8;
            let gg = (clampf(g[[0, i, j]], 0.0, 1.0) * 255.0).round() as u8;
            let bb = (clampf(b[[0, i, j]], 0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel(j as u32, i as u32, Rgb([rr, gg, bb]));
        }
    }
    img
}

/// ----- simple “Turbo-like” colormap via 5 anchors -----
fn colormap(val01: f32) -> Rgb<u8> {
    // blue -> cyan -> yellow -> orange -> red
    const C: [(f32, [u8; 3]); 5] = [
        (0.0, [18, 34, 98]),
        (0.25, [1, 135, 189]),
        (0.50, [68, 197, 87]),
        (0.75, [254, 197, 39]),
        (1.0, [220, 30, 31]),
    ];
    let x = val01.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 1 < C.len() && x > C[i + 1].0 {
        i += 1;
    }
    let (x0, c0) = C[i];
    let (x1, c1) = C[i.min(C.len() - 2) + 1];
    let t = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
    let lerp = |a: u8, b: u8| -> u8 { (a as f32 + t * (b as f32 - a as f32)).round() as u8 };
    Rgb([lerp(c0[0], c1[0]), lerp(c0[1], c1[1]), lerp(c0[2], c1[2])])
}

/// ----- 20×20 -> 224×224 bilinear upsample -----
fn bilinear_sample_20(field: &Array2<f32>, u: f32, v: f32) -> f32 {
    // u,v in [0,20) across the field
    let uu = clampf(u, 0.0, 19.999);
    let vv = clampf(v, 0.0, 19.999);
    let x0 = uu.floor() as usize;
    let y0 = vv.floor() as usize;
    let x1 = (x0 + 1).min(19);
    let y1 = (y0 + 1).min(19);
    let dx = uu - x0 as f32;
    let dy = vv - y0 as f32;
    let f00 = field[[y0, x0]];
    let f10 = field[[y0, x1]];
    let f01 = field[[y1, x0]];
    let f11 = field[[y1, x1]];
    let f0 = f00 * (1.0 - dx) + f10 * dx;
    let f1 = f01 * (1.0 - dx) + f11 * dx;
    f0 * (1.0 - dy) + f1 * dy
}
fn upsample_20_to_224(field: &Array2<f32>) -> Vec<f32> {
    let mut out = vec![0f32; 224 * 224];
    for y in 0..224 {
        for x in 0..224 {
            let u = (x as f32 / 224.0) * 20.0;
            let v = (y as f32 / 224.0) * 20.0;
            out[y * 224 + x] = bilinear_sample_20(field, u, v);
        }
    }
    out
}

/// ----- normalize a scalar field to [0,1] with robust limits -----
fn normalize01(field: &Array2<f32>, robust: bool) -> Array2<f32> {
    let vals: Vec<f32> = field.iter().copied().collect();
    let (minv, maxv) = if robust {
        let lo = percentile(vals.clone(), 0.02);
        let hi = percentile(vals, 0.98);
        (lo, if hi > lo { hi } else { lo + 1e-6 })
    } else {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in field.iter() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, if hi > lo { hi } else { lo + 1e-6 })
    };
    field.mapv(|v| ((v - minv) / (maxv - minv)).clamp(0.0, 1.0))
}

/// ----- render a 20×20 scalar field as a 224×224 heatmap PNG -----
fn save_heatmap_20(
    field: &Array2<f32>,
    out_path: &Path,
    robust_norm: bool,
) -> image::ImageResult<()> {
    let norm = normalize01(field, robust_norm);
    let up = upsample_20_to_224(&norm);
    let mut img = RgbImage::new(224, 224);
    for y in 0..224 {
        for x in 0..224 {
            let v = up[y * 224 + x];
            img.put_pixel(x as u32, y as u32, colormap(v));
        }
    }
    img.save(out_path)
}

/// ----- overlay a 224×224 heatmap on a 224×224 RGB image -----
fn overlay_on_rgb(base: &RgbImage, field: &Array2<f32>, alpha: f32, robust_norm: bool) -> RgbImage {
    let norm = normalize01(field, robust_norm);
    let up = upsample_20_to_224(&norm);
    let mut out = base.clone();
    for y in 0..224 {
        for x in 0..224 {
            let v = up[y * 224 + x];
            let Rgb([r, g, b]) = colormap(v);
            let Rgb([br, bg, bb]) = *base.get_pixel(x as u32, y as u32);
            let a = alpha.clamp(0.0, 1.0);
            let rr = (a * r as f32 + (1.0 - a) * br as f32).round() as u8;
            let gg = (a * g as f32 + (1.0 - a) * bg as f32).round() as u8;
            let bb2 = (a * b as f32 + (1.0 - a) * bb as f32).round() as u8;
            out.put_pixel(x as u32, y as u32, Rgb([rr, gg, bb2]));
        }
    }
    out
}

/// Public: dump heatmaps & overlays for the maps you have.
/// - `x224` is the [1,3,224,224] input (for overlay background)
/// - `t224`/`t112` are Tenengrad mean grids (20×20) at 224 and 112
/// - `p224`/`p112`/`pfused` are probability maps (20×20, 0..1)
/// - `w20` is the 20×20 weight map (0..1)
pub struct DebugMaps<'a> {
    pub x224: &'a Array4<f32>,
    pub t224: &'a Array2<f32>,
    pub t112: Option<&'a Array2<f32>>,
    pub p224: &'a Array2<f32>,
    pub p112: Option<&'a Array2<f32>>,
    pub pfused: &'a Array2<f32>,
    pub w20: &'a Array2<f32>,
}
pub fn dump_debug_heatmaps(out_dir: &Path, dbg: DebugMaps) -> image::ImageResult<()> {
    fs::create_dir_all(out_dir).ok();

    // Log stats
    stats("t224", dbg.t224);
    if let Some(t112) = dbg.t112 {
        stats("t112", t112);
    }
    stats("p224", dbg.p224);
    if let Some(p112) = dbg.p112 {
        stats("p112", p112);
    }
    stats("pfused", dbg.pfused);
    stats("weights", dbg.w20);

    // Save pure heatmaps (PNG)
    save_heatmap_20(
        dbg.t224,
        &out_dir.join("t224_heat.png"),
        /*robust=*/ true,
    )?;
    if let Some(t112) = dbg.t112 {
        save_heatmap_20(t112, &out_dir.join("t112_heat.png"), true)?;
    }
    save_heatmap_20(
        dbg.p224,
        &out_dir.join("p224_heat.png"),
        /*robust=*/ false,
    )?;
    if let Some(p112) = dbg.p112 {
        save_heatmap_20(p112, &out_dir.join("p112_heat.png"), false)?;
    }
    save_heatmap_20(dbg.pfused, &out_dir.join("pfused_heat.png"), false)?;
    save_heatmap_20(dbg.w20, &out_dir.join("weights_heat.png"), false)?;

    // Overlays
    let base = nchw_to_rgb_224(dbg.x224);
    overlay_on_rgb(&base, dbg.p224, 0.0, false).save(out_dir.join("image.png"))?;

    overlay_on_rgb(&base, dbg.p224, 0.55, false).save(out_dir.join("overlay_p224.png"))?;
    if let Some(p112) = dbg.p112 {
        overlay_on_rgb(&base, p112, 0.55, false).save(out_dir.join("overlay_p112.png"))?;
    }
    overlay_on_rgb(&base, dbg.pfused, 0.55, false).save(out_dir.join("overlay_pfused.png"))?;
    overlay_on_rgb(&base, dbg.w20, 0.55, false).save(out_dir.join("overlay_weights.png"))?;

    if let Some(t112) = dbg.t112 {
        overlay_on_rgb(&base, t112, 0.55, false).save(out_dir.join("overlay_t112.png"))?;
    }

    overlay_on_rgb(&base, dbg.t224, 0.55, false).save(out_dir.join("overlay_t224.png"))?;

    Ok(())
}

fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// NCHW [1,3,224,224] -> grayscale image (Luma<f32>) in [0,1]
pub fn nchw_to_gray_224(x: &Array4<f32>) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    assert_eq!(x.shape(), &[1, 3, 224, 224], "expected [1,3,224,224]");
    let (h, w) = (224u32, 224u32);
    let r = x.index_axis(Axis(1), 0);
    let g = x.index_axis(Axis(1), 1);
    let b = x.index_axis(Axis(1), 2);

    let mut buf = Vec::<f32>::with_capacity((h * w) as usize);
    for i in 0..h as usize {
        for j in 0..w as usize {
            buf.push(0.299 * r[[0, i, j]] + 0.587 * g[[0, i, j]] + 0.114 * b[[0, i, j]]);
        }
    }
    ImageBuffer::<Luma<f32>, _>::from_raw(w, h, buf).expect("bad buffer")
}

/// Exact 2× antialiased downsample (2×2 average). 224×224 -> 112×112.
fn downsample_2x_gray_f32(src: &GrayF32) -> GrayF32 {
    let (w, h) = (src.width() as usize, src.height() as usize);
    assert_eq!((w, h), (224, 224));
    let (nw, nh) = (112u32, 112u32);
    let s = src.as_raw();
    let mut out = Vec::<f32>::with_capacity((nw * nh) as usize);
    for y in (0..h).step_by(2) {
        let y1 = (y + 1).min(h - 1);
        for x in (0..w).step_by(2) {
            let x1 = (x + 1).min(w - 1);
            let i00 = y * w + x;
            let i01 = y * w + x1;
            let i10 = y1 * w + x;
            let i11 = y1 * w + x1;
            out.push(0.25 * (s[i00] + s[i01] + s[i10] + s[i11]));
        }
    }
    ImageBuffer::<Luma<f32>, _>::from_raw(nw, nh, out).unwrap()
}

fn msanity(p224: &ndarray::Array2<f32>, p112: &ndarray::Array2<f32>) {
    use ndarray::Zip;
    let mut l1 = 0f32;
    let mut linf = 0f32;
    let mut gt = 0usize;
    Zip::from(p224).and(p112).for_each(|a, b| {
        let d = (a - b).abs();
        l1 += d;
        if d > linf {
            linf = d;
        }
        if b > a {
            gt += 1;
        }
    });
    log::debug!("||p112 - p224||_1 = {l1:.6e}, L_inf = {linf:.6e}, count(p112>p224) = {gt}");
}

fn stats(name: &str, a: &ndarray::Array2<f32>) -> (f32, f32, f32, f32) {
    // (min, max, mean, median)
    let mut v: Vec<f32> = a.iter().copied().collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = v.len();
    let min = v[0];
    let max = v[n - 1];
    let mean = v.iter().sum::<f32>() / n as f32;
    let med = v[n / 2];
    log::debug!("{name}: min={min:.6e} max={max:.6e} mean={mean:.6e} med={med:.6e}");
    (min, max, mean, med)
}

/// Tenengrad (mean of G^2) aggregated to a 20×20 grid for any WxH (224 or 112).
fn tenengrad_mean_grid_20(gray: &GrayF32) -> Array2<f32> {
    const K_SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const K_SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let (w, h) = (gray.width() as usize, gray.height() as usize);
    assert!(
        (w == 224 && h == 224) || (w == 112 && h == 112),
        "expected 224 or 112"
    );

    let gx: Vec<f32> = filter3x3(gray, &K_SOBEL_X).into_raw();
    let gy: Vec<f32> = filter3x3(gray, &K_SOBEL_Y).into_raw();

    // G^2 per pixel
    let mut g2 = vec![0f32; w * h];
    for i in 0..g2.len() {
        g2[i] = gx[i] * gx[i] + gy[i] * gy[i];
    }

    // tile boundaries
    let mut rb = [0usize; 21];
    let mut cb = [0usize; 21];
    for i in 0..=20 {
        rb[i] = (i * h) / 20;
        cb[i] = (i * w) / 20;
    }
    rb[20] = h;
    cb[20] = w;

    // per-tile mean
    let mut t = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let (r0, r1) = (rb[i], rb[i + 1].max(rb[i] + 1));
            let (c0, c1) = (cb[j], cb[j + 1].max(cb[j] + 1));
            let mut sum = 0f32;
            for r in r0..r1 {
                let base = r * w;
                for c in c0..c1 {
                    sum += g2[base + c];
                }
            }
            let cnt = (r1 - r0) * (c1 - c0);
            t[[i, j]] = sum / (cnt as f32);
        }
    }
    t
}

// ---------------- multi-scale blur weights (224 + 112, fused) ----------------

/// Returns: (weights_20x20, fused_blur_prob_20x20, t224_20x20, blur_global)
pub fn blur_weights_from_nchw(
    x: &Array4<f32>,
    out_dir: Option<PathBuf>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, f32) {
    // 224 path
    let gray224 = nchw_to_gray_224(x);
    let t224 = tenengrad_mean_grid_20(&gray224);

    let p224 = t224.mapv(|t| {
        let tau = TAU_TEN_224.max(EPS_T);
        let p = (tau / (t + tau)).powf(BETA);
        (p + P_FLOOR).min(1.0)
    });

    // 112 path (2× downsample with antialias), tau scales ~ /4
    let gray112 = downsample_2x_gray_f32(&gray224);
    let t112 = tenengrad_mean_grid_20(&gray112);

    // Compute medians (robust to outliers)
    let mut v224: Vec<f32> = t224.iter().copied().collect();
    let mut v112: Vec<f32> = t112.iter().copied().collect();
    v224.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v112.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med224 = v224[v224.len() / 2].max(1e-12);
    let med112 = v112[v112.len() / 2];

    // Ratio of scales on THIS image
    let r = if med224 > 0.0 {
        (med112 / med224).clamp(0.05, 0.80)
    } else {
        0.25
    };

    // Bias so 112 is a bit stricter than naive scaling
    const BIAS112: f32 = 1.25; // try 1.2–1.4
    let tau112 = TAU_TEN_224 * r * BIAS112;

    let p112 = t112.mapv(|t| {
        let tau = tau112.max(EPS_T);
        let p = (tau / (t + tau)).powf(BETA);
        (p + P_FLOOR).min(1.0)
    });

    if log::log_enabled!(log::Level::Debug) {
        let _ = stats("p224", &p224);
        let _ = stats("p112", &p112);
        msanity(&p224, &p112);
    }

    // Fuse probabilities (take the worse = more blur)
    let mut p = Array2::<f32>::zeros((20, 20));
    ndarray::Zip::from(&mut p)
        .and(&p224)
        .and(&p112)
        .for_each(|p_elem, &a, &b| {
            *p_elem = 1.0 - (1.0 - a) * (1.0 - b);
        });

    let mean = |a: &Array2<f32>| a.sum() / (a.len() as f32);
    log::debug!(
        "mean p224={:.4}, mean p112={:.4}, mean pfused={:.4}",
        mean(&p224),
        mean(&p112),
        mean(&p)
    );

    // Weights from fused blur prob
    let mut w = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let val = 1.0 - ALPHA * p[[i, j]];
            w[[i, j]] = val.max(MIN_WEIGHT).min(1.0);
        }
    }

    if let Some(out) = out_dir {
        // We will log how long it takes to dump debug heatmaps
        let start = Instant::now();
        dump_debug_heatmaps(
            &out,
            DebugMaps {
                x224: x,
                t224: &t224,
                t112: Some(&t112),
                p224: &p224,
                p112: Some(&p112),
                pfused: &p,
                w20: &w,
            },
        )
        .unwrap();
        let duration = start.elapsed();
        log::debug!("Finished dumping debug heatmaps in {duration:?}");
    }

    let blur_global = p.sum() / 400.0;
    (w, p, t224, blur_global)
}

/// Bilinear sample of a 20x20 field at (u,v) in "cell" coordinates, where 0..20 maps across the image.
/// We clamp to [0, 19.999] to avoid going past the last cell.
fn bilinear_20(field: &Array2<f32>, u: f32, v: f32) -> f32 {
    assert_eq!(field.shape(), &[20, 20]);
    let uu = clampf(u, 0.0, 19.999);
    let vv = clampf(v, 0.0, 19.999);
    let x0 = uu.floor() as usize;
    let y0 = vv.floor() as usize;
    let x1 = (x0 + 1).min(19);
    let y1 = (y0 + 1).min(19);
    let dx = uu - x0 as f32;
    let dy = vv - y0 as f32;

    let f00 = field[[y0, x0]];
    let f10 = field[[y0, x1]];
    let f01 = field[[y1, x0]];
    let f11 = field[[y1, x1]];
    let f0 = f00 * (1.0 - dx) + f10 * dx;
    let f1 = f01 * (1.0 - dx) + f11 * dx;
    f0 * (1.0 - dy) + f1 * dy
}

/// ROI-aware mean of a 20x20 field over a bbox in original coords, using bilinear sampling.
/// We uniformly sample ROI_SAMPLES x ROI_SAMPLES points inside the bbox, map to 20x20 coords, and average.
fn roi_align_mean_20(field: &Array2<f32>, bbox: BBoxF, img_w: u32, img_h: u32) -> f32 {
    let (w, h) = (img_w as f32, img_h as f32);
    let (x0, y0, x1, y1) = (
        bbox.x0.max(0.0),
        bbox.y0.max(0.0),
        bbox.x1.min(w),
        bbox.y1.min(h),
    );
    let bw = (x1 - x0).max(1.0);
    let bh = (y1 - y0).max(1.0);

    let mut acc = 0f32;
    let mut cnt = 0usize;
    for sy in 0..ROI_SAMPLES {
        for sx in 0..ROI_SAMPLES {
            let fx = x0 + (sx as f32 + 0.5) / (ROI_SAMPLES as f32) * bw;
            let fy = y0 + (sy as f32 + 0.5) / (ROI_SAMPLES as f32) * bh;
            // map to 20x20 coordinates via 224 alignment (PaQ input size)
            let u = (fx / w) * 20.0;
            let v = (fy / h) * 20.0;
            acc += bilinear_20(field, u, v);
            cnt += 1;
        }
    }
    acc / (cnt as f32)
}

/// Estimate approx number of 224-scale cells covered by bbox (for coverage prior).
fn approx_cell_coverage_224(bbox: BBoxF, img_w: u32, img_h: u32) -> f32 {
    let (w, h) = (img_w as f32, img_h as f32);
    let bw224 = ((bbox.x1 - bbox.x0).max(0.0) / w) * 224.0;
    let bh224 = ((bbox.y1 - bbox.y0).max(0.0) / h) * 224.0;
    let cell = (224.0 / 20.0) * (224.0 / 20.0); // 11.2^2
    (bw224 * bh224 / cell).max(0.0)
}

/// Convert RgbImage crop -> grayscale Luma<f32> in [0,1]
fn rgb_crop_to_gray_f32(
    img: &RgbImage,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (x0, y0, x1, y1) = (
        x0.min(x1),
        y0.min(y1),
        x1.min(img.width()),
        y1.min(img.height()),
    );
    let w = (x1 - x0).max(1);
    let h = (y1 - y0).max(1);
    let mut buf = Vec::<f32>::with_capacity((w * h) as usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get_pixel(x, y).0;
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            buf.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    ImageBuffer::<Luma<f32>, _>::from_raw(w, h, buf).unwrap()
}

/// Native-resolution detail gate (0..1) from Tenengrad + edge density on the **un-resized** crop.
fn native_detail_probability(img: &RgbImage, bbox: BBoxF) -> (f32, u32, u32, u32, u32) {
    let (w, h) = (img.width(), img.height());
    let x0 = clampf(bbox.x0, 0.0, w as f32) as u32;
    let y0 = clampf(bbox.y0, 0.0, h as f32) as u32;
    let x1 = clampf(bbox.x1, 0.0, w as f32) as u32;
    let y1 = clampf(bbox.y1, 0.0, h as f32) as u32;

    let mut gray = rgb_crop_to_gray_f32(img, x0, y0, x1, y1);
    if GAUSS_SIGMA_NATIVE > 0.0 {
        gray = gaussian_blur_f32(&gray, GAUSS_SIGMA_NATIVE);
    }

    // Sobel on native crop
    const K_SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const K_SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let gx: Vec<f32> = filter3x3(&gray, &K_SOBEL_X).into_raw();
    let gy: Vec<f32> = filter3x3(&gray, &K_SOBEL_Y).into_raw();

    // Grad mag and Tenengrad mean
    let mut g = Vec::<f32>::with_capacity(gx.len());
    let mut ten_sum = 0f32;
    for i in 0..gx.len() {
        let m2: f32 = gx[i] * gx[i] + gy[i] * gy[i];
        g.push(m2.sqrt());
        ten_sum += m2;
    }
    let ten_mean = ten_sum / (gx.len() as f32);

    // Edge density with adaptive threshold at 0.5 * P95
    let mut gcopy = g.clone();
    let k = ((gcopy.len() as f32 - 1.0) * 0.95).round() as usize;
    gcopy.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    let p95 = gcopy[k];
    let tau_e = 0.5 * p95;
    let mut count = 0usize;
    for &val in g.iter() {
        if val > tau_e {
            count += 1;
        }
    }
    let edge_density = (count as f32) / (g.len() as f32);

    // Map to probabilities (logistics)
    let s = (x1 - x0).min(y1 - y0) as f32;
    let tau_t = TAU_TEN_224 * (S_REF / s.max(1.0)).powi(2); // smaller crops need less absolute energy
    let d_t = sigmoid(8.0 * ((ten_mean - tau_t) / tau_t));
    let d_e = sigmoid(8.0 * (edge_density - 0.05));
    (d_t.min(d_e), x0, y0, x1, y1)
}

/// Compute per-detection quality with ROI-aware pooling + native detail + priors.
pub fn detection_quality(
    q20: &Array2<f32>,   // PaQ 20x20 local map (e.g., 0..100)
    w20: &Array2<f32>,   // blur weights 20x20 (1-ALPHA*P)
    bbox: BBoxF,         // in native image pixels
    orig_img: &RgbImage, // native frame
) -> DetectionQuality {
    assert_eq!(q20.shape(), &[20, 20]);
    assert_eq!(w20.shape(), &[20, 20]);

    let (img_w, img_h) = (orig_img.width(), orig_img.height());

    let q_roi = roi_align_mean_20(q20, bbox, img_w, img_h);
    let w_roi = roi_align_mean_20(w20, bbox, img_w, img_h);

    let (detail, x0, y0, x1, y1) = native_detail_probability(orig_img, bbox);

    // size prior
    let s = ((x1 - x0).min(y1 - y0)) as f32;
    let size_prior = clampf(s / S_REF, 0.0, 1.0);

    // coverage prior (on 224 grid)
    let cov_cells = approx_cell_coverage_224(bbox, img_w, img_h);
    let coverage_prior = clampf(cov_cells / COV_REF, 0.0, 1.0);

    // final quality in the same scale as q20
    let quality = q_roi * w_roi * detail * size_prior * coverage_prior;

    DetectionQuality {
        quality,
        q_roi_mean: q_roi,
        w_roi_mean: w_roi,
        detail_prob: detail,
        size_prior,
        coverage_prior,
    }
}

/// Convenience: fuse a full-frame PaQ scalar with full-frame blur_global (if you want a single image score).
pub fn image_overall_from_paq_and_blur(paq_global: f32, blur_global: f32) -> f32 {
    let w_mean = clampf(1.0 - ALPHA * blur_global, MIN_WEIGHT, 1.0);
    paq_global * w_mean
}
