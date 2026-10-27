"""MediaPipe Face Landmarker (IMAGE mode) re-implemented with PyTorch only.

Load the checkpoint written by convert_to_pytorch.py and reproduce MediaPipe's
FaceLandmarker outputs (478 face landmarks, 52 blendshapes, facial transformation
matrix) without importing mediapipe. Every pre/post-processing step mirrors the
MediaPipe C++ calculators it replaces; the function docstrings name them.

Usage:
    python face_mesh_torch.py path/to/image.jpg
    python face_mesh_torch.py path/to/image.jpg --out-json mesh.json --num-faces 2
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.io import ImageReadMode, decode_image

DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "face_landmarker_torch.pt"

F32 = torch.float32
F64 = torch.float64

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]

# face_blendshapes_graph.cc: kLandmarksSubsetIdxs
BLENDSHAPE_LANDMARKS = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
    65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
    168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270,
    276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
    318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379,
    380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]


def f32(x) -> float:
    """Round a Python (double) scalar to float32, as a C++ `float` would hold it."""
    return torch.tensor(x, dtype=F32).item()


class _LibM:
    """Single-precision transcendentals as MediaPipe computes them: the result rounded from
    an accurate double evaluation. (The system libm's sinf is off by one ULP for some inputs
    and does not match MediaPipe, whose binary carries its own implementation.)"""

    @staticmethod
    def expf(x): return f32(math.exp(x))
    @staticmethod
    def cosf(x): return f32(math.cos(x))
    @staticmethod
    def sinf(x): return f32(math.sin(x))
    @staticmethod
    def tanf(x): return f32(math.tan(x))
    @staticmethod
    def atan2f(y, x): return f32(math.atan2(y, x))


libm = _LibM()


def expf(x):
    """float32 exp (rounded from double), elementwise."""
    return torch.exp(x.to(F64)).to(F32)


def fma(a, b, c):
    """Correctly rounded float32 fused multiply-add (NEON vfmaq_f32), computed in float64.

    a*b is exact in float64; the float64 sum is corrected with its exact rounding error so
    that the final rounding to float32 is a single correct rounding (no double rounding).
    """
    a, b, c = (torch.as_tensor(v).to(F64) for v in (a, b, c))
    p = a * b
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)  # p + c == s + err exactly
    r = s.to(F32)
    rd = r.to(F64)
    up = torch.nextafter(r, torch.full_like(r, math.inf))
    down = torch.nextafter(r, torch.full_like(r, -math.inf))
    r = torch.where(((s - rd) * 2 == up.to(F64) - rd) & (err > 0), up, r)
    return torch.where(((rd - s) * 2 == rd - down.to(F64)) & (err < 0), down, r)


# ---------------------------------------------------------------------------
# TFLite graph executor
# ---------------------------------------------------------------------------

def _activation(x, act):
    if act is None:
        return x
    if act == "RELU":
        return F.relu(x)
    if act == "RELU6":
        return x.clamp(0.0, 6.0)
    if act == "RELU_N1_TO_1":
        return x.clamp(-1.0, 1.0)
    if act == "TANH":
        return torch.tanh(x)
    raise NotImplementedError(act)


def _same_padding(size, kernel, stride, dilation):
    out = -(-size // stride)
    total = max((out - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    return total // 2, total - total // 2


# XNNPACK xnn_table_exp2minus_k_over_64: bits of 2**(-k/64), k = 0..63
_EXP2_MINUS_K_OVER_64 = [
    0x3F800000, 0x3F7F64D2, 0x3F7ECD87, 0x3F7E3A29, 0x3F7DAAC3, 0x3F7D1F62, 0x3F7C980F, 0x3F7C14D5,
    0x3F7B95C2, 0x3F7B1ADF, 0x3F7AA43A, 0x3F7A31DC, 0x3F79C3D3, 0x3F795A2B, 0x3F78F4F0, 0x3F78942D,
    0x3F7837F0, 0x3F77E046, 0x3F778D3A, 0x3F773EDA, 0x3F76F532, 0x3F76B051, 0x3F767043, 0x3F763516,
    0x3F75FED7, 0x3F75CD94, 0x3F75A15B, 0x3F757A3A, 0x3F75583F, 0x3F753B79, 0x3F7523F6, 0x3F7511C4,
    0x3F7504F3, 0x3F74FD92, 0x3F74FBAF, 0x3F74FF5B, 0x3F7508A4, 0x3F75179A, 0x3F752C4D, 0x3F7546CD,
    0x3F75672A, 0x3F758D75, 0x3F75B9BE, 0x3F75EC15, 0x3F76248C, 0x3F766334, 0x3F76A81E, 0x3F76F35B,
    0x3F7744FD, 0x3F779D16, 0x3F77FBB8, 0x3F7860F5, 0x3F78CCDF, 0x3F793F89, 0x3F79B907, 0x3F7A396A,
    0x3F7AC0C7, 0x3F7B4F30, 0x3F7BE4BA, 0x3F7C8177, 0x3F7D257D, 0x3F7DD0DF, 0x3F7E83B3, 0x3F7F3E0C,
]


def _bits_to_f32(bits):
    return bits.to(torch.int32).view(F32)


def _f32_bits(x):
    return x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _neon_recpe(d):
    """AArch64 FRECPE (8-bit reciprocal estimate, FPCR.AH = 0) for positive normal floats."""
    bits = _f32_bits(d)
    exp = (bits >> 23) & 0xFF
    scaled = 0x100 | ((bits >> 15) & 0xFF)
    estimate = ((1 << 19) // (scaled * 2 + 1) + 1) // 2
    return _bits_to_f32(((253 - exp) << 23) | ((estimate & 0xFF) << 15))


def _neon_recps(a, b):
    """AArch64 FRECPS: 2 - a*b, fused."""
    return fma(-a, b, torch.full_like(a, 2.0))


def xnnpack_sigmoid(x):
    """xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps (XNNPACK's aarch64 sigmoid)."""
    z = x.abs()
    magic_bias = torch.tensor(float.fromhex("0x1.800000p17"), dtype=F32)
    n = fma(z, torch.tensor(float.fromhex("-0x1.715476p0"), dtype=F32), magic_bias)
    n_bits = _f32_bits(n)
    table = torch.tensor(_EXP2_MINUS_K_OVER_64, dtype=torch.int64)
    s = _bits_to_f32((table[n_bits & 0x3F] + ((n_bits << 17) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    n = n - magic_bias
    t = fma(n, torch.tensor(float.fromhex("0x1.62E430p-1"), dtype=F32), z)
    p = t * torch.tensor(float.fromhex("0x1.FFFF0Ap-2"), dtype=F32)
    p = fma(-p, t, t)
    y = fma(-s, p, s)
    d = y + 1.0
    r = _neon_recpe(d)
    r = r * _neon_recps(r, d)
    r = r * _neon_recps(r, d)
    f = y * r
    f = torch.where(z > float.fromhex("0x1.5D589Ep+6"), torch.zeros_like(f), f)
    return torch.where(x < 0, f, 1.0 - f)


def _conv_taps(x, kh, kw, n):
    """NHWC input -> list of kh*kw shifted views [N, Ho, Wo, C], row-major over (ky, kx)."""
    (sh, sw), (dh, dw) = n["stride"], n["dilation"]
    if n["padding"] == "SAME":
        pt, pb = _same_padding(x.shape[1], kh, sh, dh)
        pl, pr = _same_padding(x.shape[2], kw, sw, dw)
        x = F.pad(x, (0, 0, pl, pr, pt, pb))
    ho = (x.shape[1] - (kh - 1) * dh - 1) // sh + 1
    wo = (x.shape[2] - (kw - 1) * dw - 1) // sw + 1
    return [x[:, ky * dh:ky * dh + sh * (ho - 1) + 1:sh, kx * dw:kx * dw + sw * (wo - 1) + 1:sw, :]
            for ky in range(kh) for kx in range(kw)]


def _xnnpack_conv(x, w, b, n):
    """Convolution with XNNPACK's (aarch64 NEON-FMA) accumulation order, bit-exact:
    - depthwise: bias, then one fma per tap in column-major tap order;
    - 1x1 dense (GEMM): bias, then fma over input channels. Output channels are tiled in
      blocks of 32 (64 when there are at most 4 rows); a trailing partial block no wider
      than half a tile goes through a narrower kernel that splits alternating input
      channels over two accumulators (bias in the first) and sums them at the end; the last
      K % 4 input channels go to the first accumulator;
    - kxk dense (IGEMM): bias, then fma over taps (row-major), channels within a tap."""
    out_ch, in_ch, kh, kw = w.shape
    taps = _conv_taps(x, kh, kw, n)
    shape = taps[0].shape[:3] + (out_ch,)
    acc = (b if b is not None else torch.zeros(out_ch)).expand(shape).clone()
    if n["op"] == "DEPTHWISE_CONV_2D":
        wt = w.reshape(out_ch, kh * kw)
        for kx in range(kw):
            for ky in range(kh):
                acc = fma(taps[ky * kw + kx], wt[:, ky * kw + kx], acc)
        return acc
    wt = w.permute(2, 3, 1, 0).reshape(kh * kw, in_ch, out_ch)
    if kh * kw == 1:
        x = taps[0]
        few_rows = x[..., 0].numel() <= 4
        tile = 64 if few_rows else 32
        split = out_ch // tile * tile
        if out_ch - split > tile // 2:
            split = out_ch
        main_k = in_ch - in_ch % 4
        head, tail = acc[..., :split], [acc[..., split:], torch.zeros_like(acc[..., split:])]
        for c in range(in_ch):
            xc = x[..., c:c + 1]
            head = fma(xc, wt[0, c, :split], head)
            k = c % 2 if c < main_k else 0
            tail[k] = fma(xc, wt[0, c, split:], tail[k])
        return torch.cat([head, tail[0] + tail[1]], dim=-1)
    for t, tap in enumerate(taps):
        for c in range(in_ch):
            acc = fma(tap[..., c:c + 1], wt[t, c], acc)
    return acc


class TFLiteGraph(nn.Module):
    """Executes a converted TFLite graph. Tensors keep TFLite's NHWC layout."""

    def __init__(self, spec, tensors, exact=True):
        super().__init__()
        self.spec = spec
        self.exact = exact  # reproduce XNNPACK's float arithmetic bit-for-bit (slower)
        for name, value in tensors.items():
            self.register_buffer(name, value)

    def _val(self, env, ref):
        return getattr(self, ref["const"]) if "const" in ref else env[ref["tensor"]]

    def forward(self, *inputs):
        env = dict(zip(self.spec["inputs"], inputs))
        for node in self.spec["nodes"]:
            env[node["out"]] = getattr(self, "_op_" + node["op"].lower())(env, node)
        return [env[t] for t in self.spec["outputs"]]

    def _op_conv_2d(self, env, n):
        if self.exact:
            return _activation(_xnnpack_conv(self._val(env, n["input"]), getattr(self, n["weight"]),
                                             getattr(self, n["bias"]) if n["bias"] else None, n), n["act"])
        x = self._val(env, n["input"]).permute(0, 3, 1, 2)
        w = getattr(self, n["weight"])
        b = getattr(self, n["bias"]) if n["bias"] else None
        (sh, sw), (dh, dw) = n["stride"], n["dilation"]
        if n["padding"] == "SAME":
            pt, pb = _same_padding(x.shape[2], w.shape[2], sh, dh)
            pl, pr = _same_padding(x.shape[3], w.shape[3], sw, dw)
            if pt or pb or pl or pr:
                x = F.pad(x, (pl, pr, pt, pb))
        groups = x.shape[1] if n["op"] == "DEPTHWISE_CONV_2D" else 1
        y = F.conv2d(x, w, b, stride=(sh, sw), dilation=(dh, dw), groups=groups)
        return _activation(y, n["act"]).permute(0, 2, 3, 1)

    _op_depthwise_conv_2d = _op_conv_2d

    def _op_max_pool_2d(self, env, n):
        x = self._val(env, n["input"]).permute(0, 3, 1, 2)
        (kh, kw), (sh, sw) = n["kernel"], n["stride"]
        if n["padding"] == "SAME":
            pt, pb = _same_padding(x.shape[2], kh, sh, 1)
            pl, pr = _same_padding(x.shape[3], kw, sw, 1)
            if pt or pb or pl or pr:
                x = F.pad(x, (pl, pr, pt, pb), value=-math.inf)
        y = F.max_pool2d(x, (kh, kw), (sh, sw))
        return _activation(y, n["act"]).permute(0, 2, 3, 1)

    def _op_pad(self, env, n):
        pads = [p for pair in reversed(n["paddings"]) for p in pair]
        return F.pad(self._val(env, n["input"]), pads)

    def _op_add(self, env, n):
        return _activation(self._val(env, n["a"]) + self._val(env, n["b"]), n["act"])

    def _op_sub(self, env, n):
        return _activation(self._val(env, n["a"]) - self._val(env, n["b"]), n["act"])

    def _op_mul(self, env, n):
        return _activation(self._val(env, n["a"]) * self._val(env, n["b"]), n["act"])

    def _op_div(self, env, n):
        return _activation(self._val(env, n["a"]) / self._val(env, n["b"]), n["act"])

    def _op_squared_difference(self, env, n):
        d = self._val(env, n["a"]) - self._val(env, n["b"])
        return d * d

    def _op_prelu(self, env, n):
        x, alpha = self._val(env, n["a"]), self._val(env, n["b"])
        return torch.where(x >= 0, x, x * alpha)

    def _op_relu(self, env, n):
        return F.relu(self._val(env, n["input"]))

    def _op_logistic(self, env, n):
        x = self._val(env, n["input"])
        return xnnpack_sigmoid(x) if self.exact else torch.sigmoid(x)

    def _op_neg(self, env, n):
        return -self._val(env, n["input"])

    def _op_sqrt(self, env, n):
        return torch.sqrt(self._val(env, n["input"]))

    def _op_rsqrt(self, env, n):
        return torch.rsqrt(self._val(env, n["input"]))

    def _op_reshape(self, env, n):
        return self._val(env, n["input"]).reshape(n["out_shape"])

    def _op_concatenation(self, env, n):
        return _activation(torch.cat([self._val(env, r) for r in n["inputs"]], dim=n["axis"]), n["act"])

    def _op_mean(self, env, n):
        x = self._val(env, n["input"])
        if not self.exact or len(n["axes"]) != 1:
            return x.mean(dim=n["axes"], keepdim=n["keep_dims"])
        axis = n["axes"][0] % x.dim()
        count = x.shape[axis]
        x = x.movedim(axis, -1)
        if axis == x.dim() - 1 and count % 16 == 0:
            # XNNPACK contiguous reduction: 16 running lanes, then a halving tree.
            lanes = x[..., :16]
            for i in range(16, count, 16):
                lanes = lanes + x[..., i:i + 16]
            while lanes.shape[-1] > 1:
                half = lanes.shape[-1] // 2
                lanes = lanes[..., :half] + lanes[..., half:]
            total = lanes[..., 0]
        else:
            total = x[..., 0]
            for i in range(1, count):
                total = total + x[..., i]
        out = total * f32(1.0 / count)
        return out.unsqueeze(axis) if n["keep_dims"] else out

    def _op_sum(self, env, n):
        return self._val(env, n["input"]).sum(dim=n["axes"], keepdim=n["keep_dims"])

    def _op_transpose(self, env, n):
        return self._val(env, n["input"]).permute(n["perm"])

    def _op_strided_slice(self, env, n):
        x = self._val(env, n["input"])
        index = []
        for d in range(x.dim()):
            begin, end, stride = n["begin"][d], n["end"][d], n["strides"][d]
            if n["shrink_axis_mask"] >> d & 1:
                index.append(begin)
                continue
            if stride <= 0:
                raise NotImplementedError("STRIDED_SLICE with non-positive stride")
            index.append(slice(None if n["begin_mask"] >> d & 1 else begin,
                               None if n["end_mask"] >> d & 1 else end, stride))
        return x[tuple(index)]


# ---------------------------------------------------------------------------
# Image -> tensor (ImageToTensorCalculator, OpenCV converter)
#
# MediaPipe's bundled OpenCV (a 5.x snapshot) runs the classic warp path reproduced here:
# double-precision source coordinates rounded to 1/32 pixel, then fixed-point (Q15)
# bilinear blending of uint8 pixels. Its RotatedRect::points is the OpenCV 5 formulation.
# ---------------------------------------------------------------------------

def box_points(cx, cy, w, h, angle_deg):
    """cv::RotatedRect::points / cv::boxPoints, OpenCV 5 formulation (float32)."""
    a_rad = angle_deg * math.pi / 180.0
    c, s, cx, cy, w, h = (torch.tensor(v, dtype=F32) for v in
                          (f32(math.cos(a_rad)), f32(math.sin(a_rad)), cx, cy, w, h))
    b, a = c * 0.5, s * 0.5
    ah, aw, bh, bw = a * h, a * w, b * h, b * w
    pts = [(cx - ah - bw, cy + bh - aw), (cx + ah - bw, cy - bh - aw),
           (cx + ah + bw, cy - bh + aw), (cx - ah + bw, cy + bh + aw)]
    return [(x.item(), y.item()) for x, y in pts]


def lu_solve(a, b):
    """cv::hal::LU64f: Gaussian elimination with partial pivoting, then back substitution."""
    a, b, m = [row[:] for row in a], b[:], len(a)
    for i in range(m):
        k = max(range(i, m), key=lambda j: (abs(a[j][i]), -j))
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
        d = -1.0 / a[i][i]
        for j in range(i + 1, m):
            alpha = a[j][i] * d
            for kk in range(i + 1, m):
                a[j][kk] += alpha * a[i][kk]
            b[j] += alpha * b[i]
    for i in range(m - 1, -1, -1):
        s = b[i]
        for kk in range(i + 1, m):
            s -= a[i][kk] * b[kk]
        b[i] = s / a[i][i]
    return b


def get_perspective_transform(src, dst):
    """cv::getPerspectiveTransform: 8x8 system in double (point products rounded to float)."""
    a = [[0.0] * 8 for _ in range(8)]
    b = [0.0] * 8
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src, dst)):
        a[i][0] = a[i + 4][3] = sx
        a[i][1] = a[i + 4][4] = sy
        a[i][2] = a[i + 4][5] = 1.0
        a[i][6] = -f32(sx * dx)
        a[i][7] = -f32(sy * dx)
        a[i + 4][6] = -f32(sx * dy)
        a[i + 4][7] = -f32(sy * dy)
        b[i], b[i + 4] = dx, dy
    m = lu_solve(a, b) + [1.0]
    return [m[0:3], m[3:6], m[6:9]]


def invert_3x3(m):
    """cv::invert(DECOMP_LU) closed form for 3x3 doubles."""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    r = 1.0 / det
    return [
        [(e * i - f * h) * r, (c * h - b * i) * r, (b * f - c * e) * r],
        [(f * g - d * i) * r, (a * i - c * g) * r, (c * d - a * f) * r],
        [(d * h - e * g) * r, (b * g - a * h) * r, (a * e - b * d) * r],
    ]


def _two_prod(a, b):
    p = a * b
    c = 134217729.0 * a
    ah = c - (c - a)
    c = 134217729.0 * b
    bh = c - (c - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dfma(a, b, c):
    """Fused multiply-add on float64 tensors (NEON vfmaq_f64), via error-free transforms."""
    a, b, c = (torch.as_tensor(v, dtype=F64) for v in (a, b, c))
    p, p_err = _two_prod(a, b)
    s = p + c
    bb = s - p
    s_err = (p - (s - bb)) + (c - bb)
    return s + (p_err + s_err)


def warp_perspective_linear(img, inv_m, out_w, out_h, border_zero):
    """cv::warpPerspective(INTER_LINEAR) on uint8 HxWxC (OpenCV 4 WarpPerspectiveInvoker +
    remapBilinear). inv_m maps destination pixels to source pixels."""
    src_h, src_w, _ = img.shape
    m = [float(v) for row in inv_m for v in row]
    ys = torch.arange(out_h, dtype=F64).reshape(-1, 1)
    xs = torch.arange(out_w, dtype=F64).reshape(1, -1)
    # The invoker walks 64-pixel-wide blocks; each block row starts from M*(x_block, y).
    block_w = min(1024 // min(16, out_h), out_w)
    xb = torch.div(xs, block_w, rounding_mode="floor") * block_w
    x1 = xs - xb
    x0, y0, w0 = ((m[i] * xb + m[i + 1] * ys) + m[i + 2] for i in (0, 3, 6))
    w = dfma(m[6], x1, w0)
    w = torch.where(w != 0, 32.0 / w, torch.zeros_like(w))  # INTER_TAB_SIZE / W
    fx = (dfma(m[0], x1, x0) * w).clamp(-2147483648.0, 2147483647.0)
    fy = (dfma(m[3], x1, y0) * w).clamp(-2147483648.0, 2147483647.0)
    fx_i = torch.round(fx).to(torch.int64)  # round half to even, like cvRound
    fy_i = torch.round(fy).to(torch.int64)
    sx, sy = (fx_i >> 5).clamp(-32768, 32767), (fy_i >> 5).clamp(-32768, 32767)
    tx, ty = (fx_i & 31).unsqueeze(-1), (fy_i & 31).unsqueeze(-1)

    src = img.to(torch.int64)

    def fetch(yy, xx):
        pix = src[yy.clamp(0, src_h - 1), xx.clamp(0, src_w - 1)]
        if border_zero:
            pix = pix * ((xx >= 0) & (xx < src_w) & (yy >= 0) & (yy < src_h)).unsqueeze(-1)
        return pix

    # Q15 bilinear weights: BilinearTab_i = (1 - t) and t products scaled by 2^15.
    acc = (fetch(sy, sx) * ((32 - ty) * (32 - tx)) + fetch(sy, sx + 1) * ((32 - ty) * tx)
           + fetch(sy + 1, sx) * (ty * (32 - tx)) + fetch(sy + 1, sx + 1) * (ty * tx)) * 32
    return ((acc + (1 << 14)) >> 15).clamp(0, 255).to(torch.uint8)


def image_to_tensor(img, roi, out_w, out_h, range_min, range_max, border_zero):
    """ImageToTensorCalculator CPU path: rotated ROI crop -> [1, H, W, 3] float tensor."""
    cx, cy, w, h, rotation = roi
    angle_deg = f32(f32(f32(rotation) * 180.0) / math.pi)
    src_pts = box_points(cx, cy, w, h, angle_deg)
    dst_pts = [(0.0, float(out_h)), (0.0, 0.0), (float(out_w), 0.0), (float(out_w), float(out_h))]
    inv_m = invert_3x3(get_perspective_transform(src_pts, dst_pts))
    pixels = warp_perspective_linear(img, inv_m, out_w, out_h, border_zero)
    scale = f32(f32(range_max - range_min) / 255.0)
    offset = f32(range_min)
    out = fma(pixels.to(F32), torch.tensor(scale, dtype=F32), torch.tensor(offset, dtype=F32))
    return out.unsqueeze(0)


def pad_roi(tensor_w, tensor_h, roi):
    """image_to_tensor_utils PadRoi with keep_aspect_ratio=true. Returns (roi, padding)."""
    cx, cy, w, h, rot = roi
    tensor_aspect = f32(tensor_h / tensor_w)
    roi_aspect = f32(h / w)
    if tensor_aspect > roi_aspect:
        new_w, new_h = w, f32(w * tensor_aspect)
        v_pad = f32(f32(1.0 - f32(roi_aspect / tensor_aspect)) / 2.0)
        padding = (0.0, v_pad, 0.0, v_pad)
    else:
        new_w, new_h = f32(h / tensor_aspect), h
        h_pad = f32(f32(1.0 - f32(tensor_aspect / roi_aspect)) / 2.0)
        padding = (h_pad, 0.0, h_pad, 0.0)
    return (cx, cy, new_w, new_h, rot), padding


def rotated_rect_to_image_matrix(roi, image_w, image_h):
    """GetRotatedSubRectToRectTransformMatrix: normalized ROI coords -> normalized image coords."""
    cx, cy, w, h, _ = (torch.tensor(v, dtype=F32) for v in roi)
    c, d = (torch.tensor(fn(f32(roi[4])), dtype=F32) for fn in (libm.cosf, libm.sinf))
    g, hh = torch.tensor(1.0, dtype=F32) / image_w, torch.tensor(1.0, dtype=F32) / image_h
    m = torch.zeros(16, dtype=F32)
    m[0] = w * c * g
    m[1] = -h * d * g
    m[3] = (-0.5 * w * c + 0.5 * h * d + cx) * g
    m[4] = w * d * hh
    m[5] = h * c * hh
    m[7] = (-0.5 * h * c - 0.5 * w * d + cy) * hh
    m[10] = w * g
    m[15] = 1.0
    return m


def normalize_radians(angle):
    return f32(angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi)))


# ---------------------------------------------------------------------------
# Face detector post-processing
# ---------------------------------------------------------------------------

def ssd_anchors():
    """SsdAnchorsCalculator, short-range BlazeFace defaults (fixed anchor size -> w = h = 1)."""
    strides, centers = [8, 16, 16, 16], []
    layer = 0
    while layer < len(strides):
        last = layer
        anchors_per_cell = 0
        while last < len(strides) and strides[last] == strides[layer]:
            anchors_per_cell += 2  # aspect ratio 1.0 + interpolated scale
            last += 1
        size = math.ceil(128 / strides[layer])
        for y in range(size):
            for x in range(size):
                for _ in range(anchors_per_cell):
                    centers.append(((x + 0.5) / size, (y + 0.5) / size))
        layer = last
    return torch.tensor(centers, dtype=F32)


def decode_detections(raw_boxes, raw_scores, anchors, min_score):
    """TensorsToDetectionsCalculator (XYWH order, 6 keypoints, sigmoid scores clipped at 100)."""
    raw = raw_boxes.reshape(-1, 16)
    ax, ay = anchors[:, 0], anchors[:, 1]
    xc = raw[:, 0] / 128.0 + ax
    yc = raw[:, 1] / 128.0 + ay
    w, h = raw[:, 2] / 128.0, raw[:, 3] / 128.0
    xmin, ymin = xc - w / 2.0, yc - h / 2.0
    xmax, ymax = xc + w / 2.0, yc + h / 2.0
    kps = raw[:, 4:16].reshape(-1, 6, 2) / 128.0 + anchors[:, None, :]
    scores = raw_scores.reshape(-1).clamp(-100.0, 100.0)
    scores = 1.0 / (1.0 + expf(-scores))
    boxes = torch.stack([xmin, ymin, xmax - xmin, ymax - ymin], dim=1)  # xmin, ymin, w, h
    keep = (scores >= min_score) & (boxes[:, 2] >= 0) & (boxes[:, 3] >= 0)
    return boxes[keep], kps[keep], scores[keep]


def weighted_nms(boxes, keypoints, scores, iou_threshold):
    """NonMaxSuppressionCalculator, WEIGHTED algorithm with IoU overlap.

    Scalar float32 arithmetic in MediaPipe's loop order (no fused multiply-add)."""
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    # Rectangle_f stores (min, max) corners; Area() and overlaps derive from those.
    xmax_all, ymax_all = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area_all = (xmax_all - boxes[:, 0]) * (ymax_all - boxes[:, 1])
    results = []
    while order:
        top = order[0]
        bx, by, bx1, by1, barea = boxes[top, 0], boxes[top, 1], xmax_all[top], ymax_all[top], area_all[top]
        cand, remained = [], []
        for i in order:
            ix0, iy0 = torch.maximum(boxes[i, 0], bx), torch.maximum(boxes[i, 1], by)
            ix1, iy1 = torch.minimum(xmax_all[i], bx1), torch.minimum(ymax_all[i], by1)
            similarity = torch.tensor(0.0)
            if ix0 <= ix1 and iy0 <= iy1:
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = area_all[i] + barea - inter
                if union > 0:
                    similarity = inter / union
            (cand if similarity > iou_threshold else remained).append(i)

        box, kps = boxes[top].clone(), keypoints[top].clone()
        if cand:
            total = torch.tensor(0.0)
            w_xmin = w_ymin = w_xmax = w_ymax = torch.tensor(0.0)
            w_kps = torch.zeros_like(kps)
            for i in cand:
                score = scores[i]
                total = total + score
                w_xmin = w_xmin + boxes[i, 0] * score
                w_ymin = w_ymin + boxes[i, 1] * score
                w_xmax = w_xmax + xmax_all[i] * score
                w_ymax = w_ymax + ymax_all[i] * score
                w_kps = w_kps + keypoints[i] * score
            xmin, ymin = w_xmin / total, w_ymin / total
            box = torch.stack([xmin, ymin, w_xmax / total - xmin, w_ymax / total - ymin])
            kps = w_kps / total
        results.append((box, kps, scores[top]))
        if len(remained) == len(order):
            break
        order = remained
    return results


def project_detection(box, kps, m):
    """DetectionProjectionCalculator: map letterboxed tensor coords back to the image."""
    def project(x, y):
        return x * m[0] + y * m[1] + m[3], x * m[4] + y * m[5] + m[7]

    kx, ky = project(kps[:, 0], kps[:, 1])
    xmin, ymin, w, h = box
    cx = torch.stack([xmin, xmin + w, xmin + w, xmin])
    cy = torch.stack([ymin, ymin, ymin + h, ymin + h])
    px, py = project(cx, cy)
    x0, y0 = px.min(), py.min()
    return torch.stack([x0, y0, px.max() - x0, py.max() - y0]), torch.stack([kx, ky], dim=1)


def detection_to_rect(box, kps, image_w, image_h, start_kp, end_kp):
    """DetectionsToRectsCalculator (box -> rect, rotation from two keypoints, target angle 0)."""
    xmin, ymin, w, h = box
    x0, y0 = kps[start_kp, 0] * image_w, kps[start_kp, 1] * image_h
    x1, y1 = kps[end_kp, 0] * image_w, kps[end_kp, 1] * image_h
    rotation = normalize_radians(0.0 - libm.atan2f(-(y1 - y0).item(), (x1 - x0).item()))
    return [(xmin + w / 2).item(), (ymin + h / 2).item(), w.item(), h.item(), rotation]


def transform_rect(rect, image_w, image_h, scale, square_long):
    """RectTransformationCalculator (no shift / extra rotation)."""
    cx, cy, w, h, rot = rect
    if square_long:
        long_side = max(f32(w * image_w), f32(h * image_h))
        w, h = f32(long_side / image_w), f32(long_side / image_h)
    return [cx, cy, f32(w * scale), f32(h * scale), rot]


# ---------------------------------------------------------------------------
# Face geometry (geometry_pipeline.cc + procrustes_solver.cc)
# ---------------------------------------------------------------------------

#
# MediaPipe computes this stage with Eigen in float32, and the wheel's geometry library is
# compiled with reassociation enabled, so the reduction orders below are those of the shipped
# machine code (read from its disassembly), not of Eigen's source:
#   - contiguous sums: four 4-lane accumulators over packets 0/4/8/12 mod 16, leftover packets
#     onto the first three, pairwise lane reduction, scalar tail;
#   - strided row sums: eight interleaved scalar accumulators (lane j takes elements j mod 8),
#     pairwise tree, remainder added sequentially; the first two rows of the centre of mass
#     instead use a 2-lane form summing groups of four columns as (a+b)+(c+d);
#   - 3-term coefficient products: rows 0-1 are a fused multiply-add chain, row 2 is c0+(c1+c2);
#   - the design-matrix GEMM accumulates each entry as one fused chain from zero;
#   - the 3x3 Jacobi SVD uses upstream Eigen's current makeJacobi and power-of-two scaling
#     (which is rounding-neutral and therefore omitted here).

_FLT_EPSILON = 1.1920928955078125e-07
_FLT_MIN = 1.1754943508222875e-38


def _mp_vector_sum(x):
    """Sum of a contiguous float32 array the way the shipped Eigen redux reduces it."""
    n = x.numel()
    a4, a16 = n // 4 * 4, n // 16 * 16
    if a4 == 0:
        total = x[0]
        for i in range(1, n):
            total = total + x[i]
        return total
    if a16 == 0:
        acc = x[0:4]
        if a4 >= 8:
            acc = acc + x[4:8]
        if a4 >= 12:
            acc = acc + x[8:12]
    else:
        v = [x[0:4], x[4:8], x[8:12], x[12:16]]
        for k in range(16, a16, 16):
            v = [v[i] + x[k + 4 * i:k + 4 * i + 4] for i in range(4)]
        extra = a4 - a16
        for i in range(extra // 4):
            v[i] = v[i] + x[a16 + 4 * i:a16 + 4 * i + 4]
        acc = (v[1] + v[0]) + (v[3] + v[2])
    total = (acc[0] + acc[1]) + (acc[2] + acc[3])
    for i in range(a4, n):
        total = total + x[i]
    return total


def _mp_rowsum8(row):
    """Strided row sum with eight interleaved accumulators, tree reduction, sequential tail."""
    n = row.numel()
    a8 = n // 8 * 8
    if n < 32:
        total = row[0]
        for i in range(1, n):
            total = total + row[i]
        return total
    lanes = row[0:8]
    for k in range(8, a8, 8):
        lanes = lanes + row[k:k + 8]
    total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    for i in range(a8, n):
        total = total + row[i]
    return total


def _mp_rowsum_tree4(row):
    """Row sum as the 2-lane code path does it: e0, then groups of four columns as (a+b)+(c+d)."""
    n = row.numel()
    total = row[0]
    last = (n - 1) // 4 * 4
    k = 1
    while k <= last:
        total = total + ((row[k] + row[k + 1]) + (row[k + 2] + row[k + 3]))
        k += 4
    for i in range(last + 1, n):
        total = total + row[i]
    return total


def _column_major(m):
    """Flatten a 3xN matrix in Eigen's column-major storage order."""
    return m.T.reshape(-1)


def _mp_coeff_product(a, b):
    """3x3 times 3xN coefficient product: rows 0-1 as a fused chain, row 2 as c0 + (c1 + c2)."""
    top = fma(a[0:2, 2:3], b[2:3], fma(a[0:2, 1:2], b[1:2], a[0:2, 0:1] * b[0:1]))
    bottom = a[2:3, 0:1] * b[0:1] + (a[2:3, 1:2] * b[1:2] + a[2:3, 2:3] * b[2:3])
    return torch.cat([top, bottom], 0)


def _eigen_gemm(a, b):
    """a (3xK) times b (3xK) transposed through Eigen's GEBP kernel: every entry is one
    sequential chain of fused multiply-adds over k, starting from zero."""
    acc = torch.zeros(3, 3, dtype=F32)
    for k in range(a.shape[1]):
        acc = fma(a[:, k:k + 1], b[:, k:k + 1].T, acc)
    return acc


def _det3(m):
    """Eigen's 3x3 determinant (bruteforce_det3_helper)."""
    def helper(a, b, c):
        return m[0, a] * (m[1, b] * m[2, c] - m[1, c] * m[2, b])
    return helper(0, 1, 2) - helper(1, 0, 2) + helper(2, 0, 1)


def _rotate(x, y, c, s):
    """Eigen's apply_rotation_in_the_plane scalar path: (x, y) -> (c*x + s*y, -s*x + c*y)."""
    return c * x + s * y, (-s) * x + c * y


def _rotate_rows(m, p, q, c, s):
    if c == 1.0 and s == 0.0:
        return
    m[p], m[q] = _rotate(m[p].clone(), m[q].clone(), c, s)


def _rotate_cols(m, p, q, c, s):
    if c == 1.0 and s == 0.0:
        return
    m[:, p], m[:, q] = _rotate(m[:, p].clone(), m[:, q].clone(), c, s)


def _make_jacobi(x, y, z):
    """JacobiRotation::makeJacobi(x, y, z), current upstream formulation; returns (c, s)."""
    abs_y = y.abs()
    deno = abs_y + abs_y
    if deno < _FLT_MIN:
        return torch.tensor(1.0, dtype=F32), torch.tensor(0.0, dtype=F32)
    delta = x - z
    abs_delta = delta.abs()
    large = bool(abs_delta > deno)
    ratio = (deno if large else abs_delta) / (abs_delta if large else deno)
    numerator = ratio if large else torch.tensor(1.0, dtype=F32)
    denominator = (torch.tensor(1.0, dtype=F32) if large else ratio) + torch.sqrt(ratio * ratio + 1.0)
    n = 1.0 / torch.sqrt(numerator * numerator + denominator * denominator)
    sine = numerator * n
    signed_sine = -sine if delta > 0 else sine
    return denominator * n, (-signed_sine if y < 0 else signed_sine)


def _real_2x2_jacobi_svd(w, p, q):
    """Eigen's real_2x2_jacobi_svd: returns the (c, s) pairs of j_left and j_right."""
    m00, m01, m10, m11 = w[p, p].clone(), w[p, q].clone(), w[q, p].clone(), w[q, q].clone()
    t, d = m00 + m11, m10 - m01
    if d.abs() < _FLT_MIN:
        c1, s1 = torch.tensor(1.0, dtype=F32), torch.tensor(0.0, dtype=F32)
    else:
        u = t / d
        s1 = 1.0 / torch.sqrt(1.0 + u * u)
        c1 = u * s1
    a00 = c1 * m00 + s1 * m10
    a01 = c1 * m01 + s1 * m11
    a11 = (-s1) * m01 + c1 * m11
    jr_c, jr_s = _make_jacobi(a00, a01, a11)
    return (c1 * jr_c + s1 * jr_s, s1 * jr_c - c1 * jr_s), (jr_c, jr_s)


def _jacobi_svd_3x3(m):
    """Eigen::JacobiSVD<Matrix3f>(m, ComputeFullU | ComputeFullV): returns (U, V)."""
    precision = 2.0 * _FLT_EPSILON
    w = m.clone()
    u, v = torch.eye(3, dtype=F32), torch.eye(3, dtype=F32)
    consider_as_zero = torch.tensor(_FLT_MIN, dtype=F32)
    max_diag = w.diagonal().abs().max()
    finished = False
    while not finished:
        finished = True
        threshold = torch.maximum(consider_as_zero, precision * max_diag)
        for p in range(1, 3):
            for q in range(p):
                if w[p, q].abs() > threshold or w[q, p].abs() > threshold:
                    finished = False
                    (lc, ls), (rc, rs) = _real_2x2_jacobi_svd(w, p, q)
                    _rotate_rows(w, p, q, lc, ls)       # workMatrix.applyOnTheLeft(p, q, j_left)
                    _rotate_cols(u, p, q, lc, ls)       # U.applyOnTheRight(p, q, j_left^T)
                    _rotate_cols(w, p, q, rc, -rs)      # workMatrix.applyOnTheRight(p, q, j_right)
                    _rotate_cols(v, p, q, rc, -rs)      # V.applyOnTheRight(p, q, j_right)
                    max_diag = torch.maximum(max_diag, torch.maximum(w[p, p].abs(), w[q, q].abs()))
                    threshold = torch.maximum(consider_as_zero, precision * max_diag)
    singular = []
    for i in range(3):
        a = w[i, i]
        if a < 0:
            u[:, i] = -u[:, i]
        singular.append(a.abs())
    for i in range(3):
        pos = i
        for j in range(i + 1, 3):
            if singular[j] > singular[pos]:
                pos = j
        if singular[pos] == 0:
            break
        if pos != i:
            singular[i], singular[pos] = singular[pos], singular[i]
            u[:, [i, pos]] = u[:, [pos, i]]
            v[:, [i, pos]] = v[:, [pos, i]]
    return u, v


def _optimal_rotation(design):
    """procrustes_solver.cc ComputeOptimalRotation."""
    u, v = _jacobi_svd_3x3(design)
    post, pre = u.clone(), v.T.clone()
    if _det3(post) * _det3(pre) < 0:
        post[:, 2] = -post[:, 2]
    return _mp_coeff_product(post, pre)


def solve_weighted_orthogonal_problem(sources, targets, sqrt_weights):
    """Weighted orthogonal Procrustes (scale * R, t) mapping sources (3xN) onto targets,
    following FloatPrecisionProcrustesSolver::InternalSolveWeightedOrthogonalProblem."""
    ws = sources * sqrt_weights
    wt = targets * sqrt_weights
    total_weight = _mp_vector_sum(sqrt_weights * sqrt_weights)
    tws = ws * sqrt_weights
    center = torch.stack([_mp_rowsum_tree4(tws[0]), _mp_rowsum_tree4(tws[1]), _mp_rowsum8(tws[2])]) / total_weight
    centered = ws - center[:, None] * sqrt_weights[None, :]
    rotation = _optimal_rotation(_eigen_gemm(wt, centered))
    numerator = _mp_vector_sum(_column_major(_mp_coeff_product(rotation, centered) * wt))
    denominator = _mp_vector_sum(_column_major(centered * ws))
    rs = (numerator / denominator) * rotation
    diffs = (wt - _mp_coeff_product(rs, ws)) * sqrt_weights
    translation = torch.stack([_mp_rowsum8(diffs[i]) for i in range(3)]) / total_weight
    transform = torch.eye(4, dtype=F32)
    transform[:3, :3] = rs
    transform[:3, 3] = translation
    return transform


class FaceGeometry:
    """ScreenToMetricSpaceConverter for FACE_LANDMARK_PIPELINE input, perspective camera
    (63 deg vertical FOV, near 1, far 10000), origin at the top-left corner."""

    def __init__(self, canonical, weights):
        self.canonical = canonical.to(F32).contiguous()
        self.sqrt_weights = torch.sqrt(weights.to(F32))

    def _estimate_scale(self, landmarks):
        col = solve_weighted_orthogonal_problem(self.canonical, landmarks, self.sqrt_weights)[:, 0]
        sq = col * col  # Matrix4f column norm: one packet, pairwise lane sum, sqrt
        return torch.sqrt((sq[0] + sq[1]) + (sq[2] + sq[3]))

    def pose_transform(self, landmarks_xyz, image_w, image_h):
        pts = landmarks_xyz[:468].to(F32)
        mean = pts[:, :2].mean(0)
        if torch.sqrt(((pts[:, :2] - mean) ** 2).sum(1).max()) <= 1e-3:
            return None  # IsScreenLandmarkListTooCompact

        t = lambda v: torch.tensor(v, dtype=F32)
        near = t(1.0)
        height_at_near = (2.0 * near) * libm.tanf(f32(0.5 * f32(math.pi / 180.0) * 63.0))
        width_at_near = (float(image_w) * height_at_near) / float(image_h)
        left, bottom = -0.5 * width_at_near, -0.5 * height_at_near
        x_scale, y_scale = (0.5 * width_at_near) - left, (0.5 * height_at_near) - bottom

        screen = pts.T.clone()
        screen[1] = 1.0 - screen[1]
        screen = screen * torch.stack([x_scale, y_scale, x_scale])[:, None]
        screen = screen + torch.stack([left, bottom, t(0.0)])[:, None]
        depth_offset = _mp_rowsum8(screen[2]) / 468.0

        def move_rescale_unproject_flip(m, scale):
            m = m.clone()
            m[2] = ((m[2] - depth_offset) + near) / scale
            m[0] = m[0] * m[2] / near
            m[1] = m[1] * m[2] / near
            m[2] = -m[2]
            return m

        flipped = screen.clone()
        flipped[2] = -flipped[2]
        first_scale = self._estimate_scale(flipped)
        intermediate = move_rescale_unproject_flip(screen, first_scale)
        second_scale = self._estimate_scale(intermediate)
        metric = move_rescale_unproject_flip(screen, first_scale * second_scale)
        return solve_weighted_orthogonal_problem(self.canonical, metric, self.sqrt_weights)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def load_graph(entry):
    graph = TFLiteGraph(entry["spec"], entry["tensors"])
    graph.eval()
    return graph


class FaceLandmarkerTorch(nn.Module):
    """FaceLandmarkerGraph in IMAGE mode: detector -> landmarks -> blendshapes + geometry."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, num_faces=1, min_face_detection_confidence=0.5,
                 min_face_presence_confidence=0.5, min_suppression_threshold=0.5):
        super().__init__()
        ckpt = torch.load(checkpoint, weights_only=True)
        self.detector = load_graph(ckpt["models"]["detector"])
        self.landmarker = load_graph(ckpt["models"]["landmarks"])
        self.blendshapes = load_graph(ckpt["models"]["blendshapes"])
        self.geometry = FaceGeometry(ckpt["geometry"]["canonical_landmarks"],
                                     ckpt["geometry"]["landmark_weights"])
        self.register_buffer("anchors", ssd_anchors())
        self.num_faces = num_faces
        self.min_detection = min_face_detection_confidence
        self.min_presence = min_face_presence_confidence
        self.min_suppression = min_suppression_threshold
        self.det_range = self._range(ckpt["models"]["detector"]["spec"])
        self.lmk_range = self._range(ckpt["models"]["landmarks"]["spec"])

    @staticmethod
    def _range(spec):
        mean, std = spec["normalization"]["mean"][0], spec["normalization"]["std"][0]
        return f32(f32(0.0 - mean) / std), f32(f32(255.0 - mean) / std)

    def detect_faces(self, img):
        """FaceDetectorGraph -> expanded face rects (normalized, with rotation)."""
        image_h, image_w = img.shape[:2]
        _, in_h, in_w, _ = self.detector.spec["input_shapes"][0]
        roi = (0.5 * image_w, 0.5 * image_h, float(image_w), float(image_h), 0.0)
        roi, _ = pad_roi(in_w, in_h, roi)
        matrix = rotated_rect_to_image_matrix(roi, image_w, image_h)
        tensor = image_to_tensor(img, roi, in_w, in_h, *self.det_range, border_zero=True)
        raw_boxes, raw_scores = self.detector(tensor)
        boxes, kps, scores = decode_detections(raw_boxes, raw_scores, self.anchors, self.min_detection)
        detections = weighted_nms(boxes, kps, scores, self.min_suppression)[: self.num_faces]
        rects = []
        for box, kp, score in detections:
            box, kp = project_detection(box, kp, matrix)
            rect = detection_to_rect(box, kp, image_w, image_h, start_kp=0, end_kp=1)
            rects.append(transform_rect(rect, image_w, image_h, scale=1.5, square_long=False))
        return rects

    def detect_landmarks(self, img, rect):
        """SingleFaceLandmarksDetectorGraph -> (478x3 normalized landmarks, presence score)."""
        image_h, image_w = img.shape[:2]
        _, in_h, in_w, _ = self.landmarker.spec["input_shapes"][0]
        cx, cy, w, h, rot = rect
        roi = (f32(cx * image_w), f32(cy * image_h), f32(w * image_w), f32(h * image_h), rot)
        tensor = image_to_tensor(img, roi, in_w, in_h, *self.lmk_range, border_zero=False)
        raw_landmarks, presence_logit, _ = self.landmarker(tensor)
        presence = (1.0 / (1.0 + expf(-presence_logit.reshape(())))).item()

        raw = raw_landmarks.reshape(-1, 3)
        x, y = raw[:, 0] / float(in_w), raw[:, 1] / float(in_h)
        z = raw[:, 2] / float(in_w) / 1.0
        m = rotated_rect_to_image_matrix(roi, image_w, image_h)
        # CalculateZScale: length of the projected (0,0)-(1,0) segment, squared in double.
        dx, dy = ((m[0] + m[3]) - m[3]).item(), ((m[4] + m[7]) - m[7]).item()
        z_scale = f32(math.sqrt(dx * dx + dy * dy))
        px = x * m[0] + y * m[1] + z * m[2] + m[3]
        py = x * m[4] + y * m[5] + z * m[6] + m[7]
        return torch.stack([px, py, z * z_scale], dim=1), presence

    def compute_blendshapes(self, landmarks, image_w, image_h):
        """FaceBlendshapesGraph: 146 landmark subset in pixels -> 52 scores."""
        pts = landmarks[BLENDSHAPE_LANDMARKS, :2] * torch.tensor([float(image_w), float(image_h)], dtype=F32)
        return self.blendshapes(pts.unsqueeze(0))[0].reshape(-1)

    @torch.no_grad()
    def forward(self, img):
        """img: HxWx3 uint8 RGB tensor. Returns a dict shaped like FaceLandmarkerResult."""
        image_h, image_w = img.shape[:2]
        result = {"face_landmarks": [], "face_blendshapes": [], "facial_transformation_matrixes": []}
        for rect in self.detect_faces(img):
            landmarks, presence = self.detect_landmarks(img, rect)
            if not presence > self.min_presence:
                continue
            result["face_landmarks"].append(landmarks)
            result["face_blendshapes"].append(self.compute_blendshapes(landmarks, image_w, image_h))
            matrix = self.geometry.pose_transform(landmarks, image_w, image_h)
            if matrix is not None:
                result["facial_transformation_matrixes"].append(matrix)
        return result


def load_image(path) -> torch.Tensor:
    """Decode an image file to an HxWx3 uint8 RGB tensor."""
    data = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8)
    return decode_image(data, mode=ImageReadMode.RGB).permute(1, 2, 0).contiguous()


def result_to_dict(result, width, height) -> dict:
    """Same JSON layout as face_mesh.py's result_to_dict."""
    faces = []
    for i, lms in enumerate(result["face_landmarks"]):
        pixels = lms * torch.tensor([width, height, width], dtype=F32)
        face = {"landmarks_normalized": lms.tolist(), "landmarks_pixels": pixels.tolist()}
        face["blendshapes"] = dict(zip(BLENDSHAPE_NAMES, result["face_blendshapes"][i].tolist()))
        if i < len(result["facial_transformation_matrixes"]):
            face["transformation_matrix"] = result["facial_transformation_matrixes"][i].tolist()
        faces.append(face)
    return {"image_width": width, "image_height": height, "faces": faces}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--out-json", type=Path, help="save landmarks/blendshapes as JSON")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint} (run convert_to_pytorch.py first)")
    torch.set_num_threads(1)  # the exact kernels are thousands of tiny ops; thread pools only add latency
    img = load_image(args.image)
    height, width = img.shape[:2]
    result = FaceLandmarkerTorch(args.checkpoint, num_faces=args.num_faces)(img)

    if not result["face_landmarks"]:
        print("No face detected.", file=sys.stderr)
        sys.exit(1)
    print(f"Detected {len(result['face_landmarks'])} face(s)")
    for i, lms in enumerate(result["face_landmarks"]):
        px = lms[:, :2] * torch.tensor([width, height], dtype=F32)
        x0, y0 = px.min(0).values.tolist()
        x1, y1 = px.max(0).values.tolist()
        print(f"  face {i}: bbox=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
    if args.out_json:
        args.out_json.write_text(json.dumps(result_to_dict(result, width, height), indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
