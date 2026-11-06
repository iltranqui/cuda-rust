# cvt_fp8

Behavior example for the FP8 conversions in `cuda_device::convert` (sm_89+).

PTX has no scalar `cvt.f32.e4m3` or `cvt.f32.e5m2`: FP8 is only a packed
source or destination format (`.e4m3x2`, `.e5m2x2`). Getting f32 values back
out of FP8 is a three-step path, and this example runs it on the device:

1. pack with `cvt_rn_satfinite[_relu]_{e4m3x2,e5m2x2}_f32`
2. unpack with `cvt_rn_f16x2_{e4m3x2,e5m2x2}`
3. widen with `cvt_f32x2_f16x2`

Widening FP8 to f16 to f32 is exact, so the host compares exact bits. One
launch uses values both formats represent exactly (and checks that the ReLU
packers clamp the negative lane). The other checks round-to-nearest-even
(`1.0625 -> 1.0`) and finite saturation (`1.0e6 -> 448` for e4m3, `57344`
for e5m2).

On GPUs below sm_89 the example prints `skipping:` and exits successfully.

Run with:

```bash
cargo oxide run cvt_fp8
```
