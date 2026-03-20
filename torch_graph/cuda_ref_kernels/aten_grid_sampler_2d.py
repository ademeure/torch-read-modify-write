"""Reference CUDA kernel for aten.grid_sampler_2d — bilinear interpolation."""
import torch
import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void aten_grid_sampler_2d(
    const float *input, const float *grid, float *output,
    unsigned int N, unsigned int C, unsigned int H, unsigned int W,
    unsigned int oH, unsigned int oW
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * oH * oW;
    if (idx >= total) return;
    unsigned int ow = idx % oW, oh = (idx / oW) % oH;
    unsigned int c = (idx / (oW * oH)) % C;
    unsigned int n = idx / (oW * oH * C);
    // Grid is [N, oH, oW, 2] — normalized coords in [-1, 1]
    float gx = grid[n*oH*oW*2 + oh*oW*2 + ow*2 + 0];
    float gy = grid[n*oH*oW*2 + oh*oW*2 + ow*2 + 1];
    // Unnormalize to pixel coords
    float ix = ((gx + 1.0f) * (float)W - 1.0f) * 0.5f;
    float iy = ((gy + 1.0f) * (float)H - 1.0f) * 0.5f;
    int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
    float xf = ix - x0, yf = iy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    float v00 = (x0>=0&&x0<(int)W&&y0>=0&&y0<(int)H) ? input[n*C*H*W+c*H*W+y0*W+x0] : 0.0f;
    float v01 = (x1>=0&&x1<(int)W&&y0>=0&&y0<(int)H) ? input[n*C*H*W+c*H*W+y0*W+x1] : 0.0f;
    float v10 = (x0>=0&&x0<(int)W&&y1>=0&&y1<(int)H) ? input[n*C*H*W+c*H*W+y1*W+x0] : 0.0f;
    float v11 = (x1>=0&&x1<(int)W&&y1>=0&&y1<(int)H) ? input[n*C*H*W+c*H*W+y1*W+x1] : 0.0f;
    output[idx] = (1-yf)*(1-xf)*v00 + (1-yf)*xf*v01 + yf*(1-xf)*v10 + yf*xf*v11;
}
"""

NN, CC, HH, WW, OH, OW = 1, 4, 8, 8, 4, 4

def init_once():
    x = torch.randn(NN, CC, HH, WW, device="cuda")
    grid = torch.randn(NN, OH, OW, 2, device="cuda") * 0.5  # keep in reasonable range
    total = NN * CC * OH * OW
    return {"kernel_source": KERNEL_SRC, "inputs": [x.contiguous(), grid.contiguous()],
            "expected": [torch.ops.aten.grid_sampler_2d.default(x, grid, 0, 0, False).flatten()],
            "outputs": ["float32;n=%d" % total], "grid": ((total + 255) // 256,), "atol": 1e-4}

def run(inputs, kernel):
    total = NN * CC * OH * OW
    return [kernel(*inputs, params=[
        kernel.in_ptr(0), kernel.in_ptr(1), kernel.out_ptr(0),
        np.uint32(NN), np.uint32(CC), np.uint32(HH), np.uint32(WW),
        np.uint32(OH), np.uint32(OW),
    ])]
