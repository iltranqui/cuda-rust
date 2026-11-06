/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! FP8 conversion round trip end-to-end example (sm_89+).
//!
//! PTX has no scalar `cvt.f32.e4m3` / `cvt.f32.e5m2`; FP8 is only a packed
//! source or destination format (`.e4m3x2` / `.e5m2x2`). The f32 round trip
//! is therefore three steps, all exercised here on the device:
//!
//!   1. pack   `cvt_rn_satfinite[_relu]_{e4m3x2,e5m2x2}_f32(lo, hi) -> u16`
//!   2. unpack `cvt_rn_f16x2_{e4m3x2,e5m2x2}(u16) -> u32` (packed f16x2)
//!   3. widen  `cvt_f32x2_f16x2(u32) -> (f32, f32)`
//!
//! Widening FP8 -> f16 -> f32 is exact, so the host compares exact bits.
//! Two launches pin down both halves of each pair:
//!
//!   - `(1.5, -0.375)` is exactly representable in both formats, so it
//!     round-trips unchanged; the ReLU packers clamp `-0.375` to `0.0`.
//!   - `(1.0625, 1.0e6)` checks rounding and saturation. `1.0625` is a tie
//!     between `1.0` and `1.125` in e4m3 (and sits between `1.0` and `1.25`
//!     in e5m2), so round-to-nearest-even must give `1.0` in both. `1.0e6`
//!     is out of range, so `satfinite` must give the largest finite value:
//!     `448` for e4m3 and `57344` for e5m2.
//!
//! Run: cargo oxide run cvt_fp8

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::convert::{
    cvt_f32x2_f16x2, cvt_rn_f16x2_e4m3x2, cvt_rn_f16x2_e5m2x2, cvt_rn_satfinite_e4m3x2_f32,
    cvt_rn_satfinite_e5m2x2_f32, cvt_rn_satfinite_relu_e4m3x2_f32,
    cvt_rn_satfinite_relu_e5m2x2_f32,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Four packers, two f32 lanes each.
const NUM_OUT: usize = 8;

// =============================================================================
// KERNEL
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 packs `(lo, hi)` to FP8 with all four packers, unpacks each
    /// word to f16x2, widens to f32, and writes:
    ///
    ///   out[0..2] = e4m3 round trip
    ///   out[2..4] = e5m2 round trip
    ///   out[4..6] = e4m3 round trip through the ReLU packer
    ///   out[6..8] = e5m2 round trip through the ReLU packer
    #[kernel]
    pub fn cvt_fp8_roundtrip(lo: f32, hi: f32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if idx.get() == 0 {
            let (e4m3_lo, e4m3_hi) =
                cvt_f32x2_f16x2(cvt_rn_f16x2_e4m3x2(cvt_rn_satfinite_e4m3x2_f32(lo, hi)));
            let (e5m2_lo, e5m2_hi) =
                cvt_f32x2_f16x2(cvt_rn_f16x2_e5m2x2(cvt_rn_satfinite_e5m2x2_f32(lo, hi)));
            let (e4m3_relu_lo, e4m3_relu_hi) = cvt_f32x2_f16x2(cvt_rn_f16x2_e4m3x2(
                cvt_rn_satfinite_relu_e4m3x2_f32(lo, hi),
            ));
            let (e5m2_relu_lo, e5m2_relu_hi) = cvt_f32x2_f16x2(cvt_rn_f16x2_e5m2x2(
                cvt_rn_satfinite_relu_e5m2x2_f32(lo, hi),
            ));
            unsafe {
                *out.get_unchecked_mut(0) = e4m3_lo;
                *out.get_unchecked_mut(1) = e4m3_hi;
                *out.get_unchecked_mut(2) = e5m2_lo;
                *out.get_unchecked_mut(3) = e5m2_hi;
                *out.get_unchecked_mut(4) = e4m3_relu_lo;
                *out.get_unchecked_mut(5) = e4m3_relu_hi;
                *out.get_unchecked_mut(6) = e5m2_relu_lo;
                *out.get_unchecked_mut(7) = e5m2_relu_hi;
            }
        }
    }
}

// =============================================================================
// HOST VERIFICATION
// =============================================================================

const LABELS: [&str; NUM_OUT] = [
    "e4m3 lo",
    "e4m3 hi",
    "e5m2 lo",
    "e5m2 hi",
    "e4m3 relu lo",
    "e4m3 relu hi",
    "e5m2 relu lo",
    "e5m2 relu hi",
];

/// Launches the kernel on `(lo, hi)` and compares against `expected` by bits.
/// Returns the number of mismatches.
fn run_case(
    name: &str,
    module: &kernels::LoadedModule,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    lo: f32,
    hi: f32,
    expected: [f32; NUM_OUT],
) -> usize {
    println!("--- {name}: (lo, hi) = ({lo}, {hi}) ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, NUM_OUT).unwrap();
    let cfg = LaunchConfig::for_num_elems(1);
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cvt_fp8_roundtrip(stream, cfg, lo, hi, &mut out_dev) }
        .expect("Kernel launch failed");
    let got = out_dev.to_host_vec(stream).unwrap();
    assert_eq!(got.len(), NUM_OUT);

    let mut failures = 0;
    for i in 0..NUM_OUT {
        if got[i].to_bits() == expected[i].to_bits() {
            println!("[{i}] {}: PASS ({})", LABELS[i], got[i]);
        } else {
            eprintln!(
                "[{i}] {}: FAIL, expected {}, got {}",
                LABELS[i], expected[i], got[i]
            );
            failures += 1;
        }
    }
    failures
}

fn main() {
    println!("=== FP8 conversion round trip (sm_89+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // FP8 conversions were introduced on Ada Lovelace (sm_89).
    if (major, minor) < (8, 9) {
        println!("\nskipping: FP8 conversions require sm_89+ (Ada Lovelace)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Exactly representable in e4m3 and e5m2: unchanged by the round trip;
    // the ReLU packers clamp the negative lane to zero.
    let exact = run_case(
        "exact values",
        &module,
        &stream,
        1.5,
        -0.375,
        [1.5, -0.375, 1.5, -0.375, 1.5, 0.0, 1.5, 0.0],
    );

    // Rounding (tie to even) and finite saturation. ReLU changes nothing
    // because both inputs are positive.
    let rounded = run_case(
        "rounding and saturation",
        &module,
        &stream,
        1.0625,
        1.0e6,
        [1.0, 448.0, 1.0, 57344.0, 1.0, 448.0, 1.0, 57344.0],
    );

    println!();
    let failures = exact + rounded;
    if failures == 0 {
        println!("SUCCESS: all FP8 pack -> unpack -> widen round trips produced correct results");
    } else {
        eprintln!("{failures} check(s) failed");
        std::process::exit(1);
    }
}
