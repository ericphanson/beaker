use image::{ImageBuffer, Luma};
use imageproc::filter::filter3x3;
use ndarray::{Array2, Array4, Axis};

const ALPHA: f32 = 0.7; // downweight strength
const K_SIGMOID: f32 = 6.0;
const TAU_TEN: f32 = 0.02; // absolute Tenengrad threshold for [0,1] pixels @224x224
const MIN_WEIGHT: f32 = 0.2;

const K_SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
const K_SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

type GrayF32 = ImageBuffer<Luma<f32>, Vec<f32>>;

fn nchw_to_gray_image(x: &Array4<f32>) -> GrayF32 {
    assert_eq!(x.shape(), &[1, 3, 224, 224]);
    let mut buf = Vec::<f32>::with_capacity(224 * 224);
    let r = x.index_axis(Axis(1), 0);
    let g = x.index_axis(Axis(1), 1);
    let b = x.index_axis(Axis(1), 2);
    for i in 0..224 {
        for j in 0..224 {
            buf.push(0.299 * r[[0, i, j]] + 0.587 * g[[0, i, j]] + 0.114 * b[[0, i, j]]);
        }
    }
    ImageBuffer::<Luma<f32>, _>::from_raw(224, 224, buf).unwrap()
}

/// Tenengrad energy map (mean of G^2 per tile), then absolute mapping -> weights
pub fn blur_weights_from_nchw(x: &Array4<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, f32) {
    let gray = nchw_to_gray_image(x);
    let gx = filter3x3(&gray, &K_SOBEL_X);
    let gy = filter3x3(&gray, &K_SOBEL_Y);
    let gx: Vec<f32> = gx.into_raw();
    let gy: Vec<f32> = gy.into_raw();

    // G^2 per pixel
    let mut g2 = vec![0f32; 224 * 224];
    for i in 0..g2.len() {
        g2[i] = gx[i] * gx[i] + gy[i] * gy[i];
    }

    // Tile bounds (exact coverage)
    let mut rb = [0usize; 21];
    let mut cb = [0usize; 21];
    for i in 0..=20 {
        rb[i] = (i * 224) / 20;
        cb[i] = (i * 224) / 20;
    }
    rb[20] = 224;
    cb[20] = 224;

    // Per-tile Tenengrad mean
    let mut t = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let (r0, r1) = (rb[i], rb[i + 1].max(rb[i] + 1));
            let (c0, c1) = (cb[j], cb[j + 1].max(cb[j] + 1));
            let mut sum = 0f32;
            for r in r0..r1 {
                let base = r * 224;
                for c in c0..c1 {
                    sum += g2[base + c];
                }
            }
            let cnt = (r1 - r0) * (c1 - c0);
            t[[i, j]] = sum / (cnt as f32);
        }
    }

    // Absolute mapping: lower Tenengrad => higher blur probability
    let mut p = Array2::<f32>::zeros((20, 20));
    for i in 0..20 {
        for j in 0..20 {
            let z = (TAU_TEN - t[[i, j]]) / TAU_TEN;
            p[[i, j]] = 1.0 / (1.0 + (-K_SIGMOID * z).exp());
        }
    }

    // Weights: downweight blurry tiles
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

    let blur_global = p.sum() / 400.0;
    (w, p, t, blur_global)
}
