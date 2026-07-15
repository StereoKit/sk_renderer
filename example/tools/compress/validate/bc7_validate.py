# BC7 mode-6 CPU reference decoder + PSNR vs the source image.
# Decode chain implemented straight from the D3D spec, independent of the
# shader. PSNR convention matches tex_psnr: 8-bit values, RGB channels.
# Usage: bc7_validate.py <source.png/jpg> <bc7_output.dds>
import struct, sys
import numpy as np
from PIL import Image

W4 = [0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64]

class BitReader:
    def __init__(self, block16):
        self.v = int.from_bytes(block16, 'little')
        self.pos = 0
    def get(self, n):
        r = (self.v >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return r

def decode_block_mode6(block16):
    br = BitReader(block16)
    mode = 0
    while mode < 8 and br.get(1) == 0:
        mode += 1
    assert mode == 6, f"not mode 6: mode {mode}"
    # R0 R1 G0 G1 B0 B1 A0 A1 (7 bits each), P0 P1
    e = [br.get(7) for _ in range(8)]
    p0, p1 = br.get(1), br.get(1)
    e0 = [(e[0] << 1) | p0, (e[2] << 1) | p0, (e[4] << 1) | p0, (e[6] << 1) | p0]
    e1 = [(e[1] << 1) | p1, (e[3] << 1) | p1, (e[5] << 1) | p1, (e[7] << 1) | p1]
    idx = [br.get(3)]  # anchor
    for _ in range(15):
        idx.append(br.get(4))
    out = np.zeros((16, 4), dtype=np.uint8)
    for px in range(16):
        w = W4[idx[px]]
        for c in range(4):
            out[px, c] = (e0[c] * (64 - w) + e1[c] * w + 32) >> 6
    return out, idx

def main(src_path, dds_path):
    src = np.array(Image.open(src_path).convert('RGBA'), dtype=np.uint8)
    h, w = src.shape[:2]

    with open(dds_path, 'rb') as f:
        dds = f.read()
    assert dds[:4] == b'DDS ' and struct.unpack_from('<I', dds, 84)[0] == 0x30315844
    dh, dw = struct.unpack_from('<II', dds, 12)
    dxgi = struct.unpack_from('<I', dds, 128)[0]
    assert (dh, dw) == (h, w) and dxgi in (98, 99), (dh, dw, dxgi)
    blocks = dds[148:]

    bw, bh = (w + 3) // 4, (h + 3) // 4
    dec = np.zeros((h, w, 4), dtype=np.uint8)
    anchor_bad = 0
    for by in range(bh):
        for bx in range(bw):
            blk = blocks[(by * bw + bx) * 16:(by * bw + bx) * 16 + 16]
            texels, idx = decode_block_mode6(blk)
            if idx[0] >= 8: anchor_bad += 1
            for px in range(16):
                x, y = bx * 4 + px % 4, by * 4 + px // 4
                if x < w and y < h:
                    dec[y, x] = texels[px]

    d = dec.astype(np.float64) - src.astype(np.float64)
    mse_rgb = np.mean(d[..., :3] ** 2)
    mse_a   = np.mean(d[..., 3] ** 2)
    # premultiplied compare, matching validate/metrics.c
    aa = src[..., 3:4].astype(np.float64) / 255.0
    ba = dec[..., 3:4].astype(np.float64) / 255.0
    dpm = src[..., :3] * aa - dec[..., :3] * ba
    mse_pm = np.mean(dpm ** 2)
    def psnr(mse): return 999.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"{w}x{h}, {bw*bh} blocks, all mode 6; anchor violations: {anchor_bad} (must be 0)")
    print(f"PSNR(RGB) {psnr(mse_rgb):.2f} dB   PSNR(A) {psnr(mse_a):.2f} dB   PSNR(PM) {psnr(mse_pm):.2f} dB")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
