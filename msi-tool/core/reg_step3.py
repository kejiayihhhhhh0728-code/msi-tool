"""
step3_least_squares_nonrigid.py
---------------------------------
Thin-plate spline (TPS) based non-rigid registration solved with
regularized least squares. Provides utilities to fit TPS from a set
of control points and build a remap (map_x, map_y) usable with
`cv2.remap` to warp images/masks.

API (main functions):
 - `fit_tps(src_pts, dst_pts, reg)` -> params
 - `tps_map(params, ctrl_pts, xs, ys)` -> mapped coordinates
 - `build_tps_remap(ctrl_src, ctrl_dst, dst_shape, reg)` -> map_x, map_y

Notes:
 - This module avoids SciPy dependency by using NumPy linear solves.
 - Regularization (`reg`) stabilizes the linear system and acts as
   a smoothing term to avoid overfitting noisy correspondences.
 - For image warping, we fit TPS mapping from target->source (inverse
   warp) so `cv2.remap` can sample from source image.

"""

import numpy as np
import cv2


def _tps_kernel(r2):
    """TPS radial basis: r^2 * log(r^2). Handle r==0 safely."""
    # r2 = squared distance
    # use where to avoid log(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = r2 * np.log(r2 + 1e-20)
    k[r2 == 0] = 0.0
    return k


def fit_tps(src_pts, dst_pts, reg=1e-3):
    """Fit TPS mapping that maps `src_pts` -> `dst_pts`.

    src_pts: (n,2) control points
    dst_pts: (n,2) target points
    reg: regularization weight added to K diagonal

    Returns dict with keys: 'ctrl': src_pts, 'w': (n,2), 'a': (3,2)
    such that f(x) = a0 + a1*x + a2*y + sum_j w_j * U(||x-ctrl_j||)
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n = src_pts.shape[0]
    if n < 3:
        raise ValueError('At least 3 control points required for TPS.')

    # Pairwise squared distances
    d2 = np.sum((src_pts[:, None, :] - src_pts[None, :, :]) ** 2, axis=2)
    K = _tps_kernel(d2)

    P = np.concatenate([np.ones((n, 1)), src_pts], axis=1)  # (n,3)

    # Assemble linear system
    A = np.zeros((n + 3, n + 3), dtype=np.float64)
    A[:n, :n] = K + reg * np.eye(n)
    A[:n, n:] = P
    A[n:, :n] = P.T

    # Right-hand side
    V = np.zeros((n + 3, 2), dtype=np.float64)
    V[:n, :] = dst_pts

    # Solve
    sol = np.linalg.solve(A, V)
    w = sol[:n, :]      # weights (n,2)
    a = sol[n:, :]      # affine (3,2)

    return {'ctrl': src_pts, 'w': w, 'a': a}


def tps_map(params, xs, ys):
    """Apply TPS mapping to points (xs, ys).

    xs, ys can be arrays of same shape. Returns (x_mapped, y_mapped).
    """
    ctrl = params['ctrl']
    w = params['w']
    a = params['a']

    # flatten inputs
    xs_f = np.asarray(xs).ravel()
    ys_f = np.asarray(ys).ravel()
    pts = np.stack([xs_f, ys_f], axis=1)  # (m,2)

    # Compute pairwise squared distance between pts and ctrl
    d2 = np.sum((pts[:, None, :] - ctrl[None, :, :]) ** 2, axis=2)  # (m,n)
    K = _tps_kernel(d2)  # (m,n)

    mapped = np.dot(K, w) + np.dot(np.concatenate([np.ones((pts.shape[0], 1)), pts], axis=1), a)
    xm = mapped[:, 0].reshape(xs.shape)
    ym = mapped[:, 1].reshape(ys.shape)
    return xm, ym


def build_tps_remap(ctrl_src, ctrl_dst, dst_shape, reg=1e-3, dtype=np.float32, chunk_rows=256):
    """Build `map_x`, `map_y` for `cv2.remap` that maps pixels in
    destination image space back to source image coordinates.

    This implementation computes the mapping in row-wise chunks to
    avoid allocating a giant (H*W, N_ctrl) temporary array.

    ctrl_src: (n,2) source control points (e.g., MSI coords)
    ctrl_dst: (n,2) destination control points (e.g., HE coords)
    dst_shape: (h, w)
    reg: TPS regularization
    chunk_rows: number of image rows processed per chunk (tune to memory)

    Returns map_x, map_y as float arrays shaped (h, w).
    """
    h, w = dst_shape[:2]
    params = fit_tps(ctrl_dst, ctrl_src, reg=reg)

    map_x = np.empty((h, w), dtype=dtype)
    map_y = np.empty((h, w), dtype=dtype)

    # Process in chunks of rows to limit memory use
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        ys = np.arange(y0, y1, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float64), ys)
        mx, my = tps_map(params, grid_x, grid_y)
        map_x[y0:y1, :] = mx.astype(dtype)
        map_y[y0:y1, :] = my.astype(dtype)

    return map_x, map_y


def warp_image_with_remap(src_img, map_x, map_y, interp=cv2.INTER_LINEAR):
    """Warp `src_img` using precomputed remap arrays."""
    # Ensure maps have same shape as desired destination
    warped = cv2.remap(src_img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


if __name__ == '__main__':
    # Quick sanity check demo using synthetic points (not a full test).
    # Creates a simple warp and applies to a small mask.
    import matplotlib.pyplot as plt

    # synthetic control points (square -> slightly bent)
    src = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64)
    dst = np.array([[12, 8], [88, 12], [92, 92], [8, 88]], dtype=np.float64)

    # simple source image
    src_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(src_img, (50, 50), 30, 255, -1)

    map_x, map_y = build_tps_remap(src, dst, dst_shape=src_img.shape, reg=1e-2)
    warped = warp_image_with_remap(src_img, map_x, map_y)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1); plt.imshow(src_img, cmap='gray'); plt.title('src')
    plt.subplot(1, 3, 2); plt.imshow(warped, cmap='gray'); plt.title('warped')
    plt.subplot(1, 3, 3); plt.scatter(dst[:, 0], dst[:, 1], c='r'); plt.scatter(src[:, 0], src[:, 1], c='b'); plt.gca().invert_yaxis(); plt.title('ctrl')
    plt.tight_layout()
    plt.show()
