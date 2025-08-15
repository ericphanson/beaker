use ndarray::{Array2, Array4, Axis};
use std::cmp::{max, min};

const EPS: f32 = 1e-6;
const ALPHA: f32 = 0.6; // strength of downweighting [0..1]
const K_SIGMOID: f32 = 8.0; // blur sensitivity (higher => harsher)
const MIN_WEIGHT: f32 = 0.2; // floor to avoid killing cells

/// Convert [1,3,224,224] NCHW -> grayscale [224,224] in [0,1]
pub fn nchw_to_gray_224(x: &Array4<f32>) -> Array2<f32> {
    assert_eq!(x.shape(), &[1, 3, 224, 224]);
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    assert_eq!(n, 1);
    assert_eq!(c, 3);
    let r = x.index_axis(Axis(1), 0);
    let g = x.index_axis(Axis(1), 1);
    let b = x.index_axis(Axis(1), 2);
    // ITU-R BT.601 luma
    let mut y = Array2::<f32>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let yy = 0.299 * r[[0, i, j]] + 0.587 * g[[0, i, j]] + 0.114 * b[[0, i, j]];
            y[[i, j]] = yy;
        }
    }
    // If your model feeds 0..255, normalize here:
    // y.mapv_inplace(|v| (v / 255.0).clamp(0.0, 1.0));
    y
}

/// Compute SML and Sobel magnitude maps on grayscale [224,224].
fn sml_and_sobel(y: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = (y.shape()[0], y.shape()[1]);
    assert_eq!((h, w), (224, 224));

    let mut sml = Array2::<f32>::zeros((h, w));
    let mut sob = Array2::<f32>::zeros((h, w));

    // Helper for clamped indexing
    let at = |r: isize, c: isize| -> f32 {
        let rr = min(max(r, 0), (h as isize) - 1) as usize;
        let cc = min(max(c, 0), (w as isize) - 1) as usize;
        y[[rr, cc]]
    };

    for r in 0..h as isize {
        for c in 0..w as isize {
            let yc = at(r, c);
            // Sum-Modified Laplacian (separable 2nd derivatives, abs)
            let lxx = (at(r, c + 1) - 2.0 * yc + at(r, c - 1)).abs();
            let lyy = (at(r + 1, c) - 2.0 * yc + at(r - 1, c)).abs();
            sml[[r as usize, c as usize]] = lxx + lyy;

            // Sobel gradients
            let gx = -at(r - 1, c - 1) + at(r - 1, c + 1) - 2.0 * at(r, c - 1) + 2.0 * at(r, c + 1)
                - at(r + 1, c - 1)
                + at(r + 1, c + 1);
            let gy = -at(r - 1, c - 1) - 2.0 * at(r - 1, c) - at(r - 1, c + 1)
                + at(r + 1, c - 1)
                + 2.0 * at(r + 1, c)
                + at(r + 1, c + 1);
            sob[[r as usize, c as usize]] = (gx * gx + gy * gy).sqrt();
        }
    }
    (sml, sob)
}

/// Compute 20x20 per-cell blur scores B, blur probabilities P, and weights W = 1 - alpha*P
pub fn blur_weights_from_nchw(x: &Array4<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, f32) {
    let y = nchw_to_gray_224(x);
    let (sml, sob) = sml_and_sobel(&y);

    // Tile boundaries: exact coverage with floor splits
    let mut rb = [0usize; 21];
    let mut cb = [0usize; 21];
    for i in 0..=20 {
        rb[i] = (i * 224) / 20;
        cb[i] = (i * 224) / 20;
    }
    rb[20] = 224;
    cb[20] = 224;

    // Per-cell blur score: B = sum(SML) / (sum(Sobel) + kappa)
    let mut b = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let (r0, r1) = (rb[i], rb[i + 1].max(rb[i] + 1));
            let (c0, c1) = (cb[j], cb[j + 1].max(cb[j] + 1));
            let mut sum_sml = 0.0f32;
            let mut sum_sob = 0.0f32;
            for r in r0..r1 {
                for c in c0..c1 {
                    sum_sml += sml[[r, c]];
                    sum_sob += sob[[r, c]];
                }
            }
            let area = ((r1 - r0) * (c1 - c0)) as f32;
            let kappa = 1e-3 * area;
            b[[i, j]] = sum_sml / (sum_sob + kappa);
        }
    }

    // Convert B -> blur probability P using per-image median (lower B => blurrier)
    let mut flat = b.iter().copied().collect::<Vec<_>>();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = flat[flat.len() / 2].max(EPS);
    let mut p = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let z = (med - b[[i, j]]) / (med + EPS); // >0 when blurrier than median
            p[[i, j]] = 1.0 / (1.0 + (-K_SIGMOID * z).exp());
        }
    }

    // Weights W = 1 - alpha*P, clamped to [MIN_WEIGHT, 1]
    let mut w = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let mut val = 1.0 - ALPHA * p[[i, j]];
            if val < MIN_WEIGHT {
                val = MIN_WEIGHT;
            }
            if val > 1.0 {
                val = 1.0;
            }
            w[[i, j]] = val;
        }
    }

    // Global blur indicator (0..1)
    let blur_global = p.sum() / 400.0;

    (w, p, b, blur_global)
}
