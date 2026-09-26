# One-axis motion compensation

Dynamic reflections can corrupt whole-image registration, so motion is estimated only in user-selected Motion ROIs placed on stable object features. Analysis ROIs are separate.

Each Motion ROI uses a Hanning-windowed phase correlation after optional raw, Gaussian, or gradient preprocessing. Low texture, low response, non-finite values, or implausibly large shifts are rejected. Valid selected-axis shifts are fused by median; the application exposes every latest ROI result, the valid count, fused shift, and quality. The pure motion core also provides a cumulative-position state for experiments that require it.

For a historical frame at cumulative position `P_i` and current frame at `P_n`, translation aligns the historical image to current coordinates. The same affine transform is applied to an all-valid mask using nearest-neighbor interpolation and constant invalid borders. Multi-frame results must use the intersection mask so padding cannot become a false defect. An optional common-valid crop is explicit.

ECC translation is experimental. Arbitrary affine and homography registration are deliberately outside V01.
