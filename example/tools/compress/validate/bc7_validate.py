# BC7 mode-6 CPU reference decoder + PSNR vs the source image.
# Decode chain implemented straight from the D3D spec, independent of the
# shader. PSNR convention matches tex_psnr: 8-bit values, RGB channels.
# Usage: bc7_validate.py <source.png/jpg> <bc7_output.dds>
import struct, sys
import numpy as np
from PIL import Image

W4 = [0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64]
W2 = [0,21,43,64]

class BitReader:
    def __init__(self, block16):
        self.v = int.from_bytes(block16, 'little')
        self.pos = 0
    def get(self, n):
        r = (self.v >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return r

def decode_block(block16):
    """Returns (texels, anchor_ok, mode)."""
    br = BitReader(block16)
    mode = 0
    while mode < 8 and br.get(1) == 0:
        mode += 1
    if mode == 6:
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
        return out, idx[0] < 8, mode
    if mode == 5:
        rot = br.get(2)
        e = [br.get(7) for _ in range(6)]     # R0 R1 G0 G1 B0 B1
        a0, a1 = br.get(8), br.get(8)
        c0 = [(v << 1) | (v >> 6) for v in (e[0], e[2], e[4])]
        c1 = [(v << 1) | (v >> 6) for v in (e[1], e[3], e[5])]
        cidx = [br.get(1)] + [br.get(2) for _ in range(15)]
        aidx = [br.get(1)] + [br.get(2) for _ in range(15)]
        out = np.zeros((16, 4), dtype=np.uint8)
        for px in range(16):
            wc, wa = W2[cidx[px]], W2[aidx[px]]
            for c in range(3):
                out[px, c] = (c0[c] * (64 - wc) + c1[c] * wc + 32) >> 6
            out[px, 3] = (a0 * (64 - wa) + a1 * wa + 32) >> 6
            if rot:  # rotation swaps alpha with channel rot-1
                ch = rot - 1
                out[px, ch], out[px, 3] = out[px, 3], out[px, ch]
        return out, cidx[0] < 2 and aidx[0] < 2, mode
    raise AssertionError(f"unexpected mode {mode}")

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
    mode_counts = {}
    for by in range(bh):
        for bx in range(bw):
            blk = blocks[(by * bw + bx) * 16:(by * bw + bx) * 16 + 16]
            texels, anchor_ok, mode = decode_block(blk)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if not anchor_ok: anchor_bad += 1
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
    mix = ", ".join(f"mode {m}: {n}" for m, n in sorted(mode_counts.items()))
    print(f"{w}x{h}, {bw*bh} blocks ({mix}); anchor violations: {anchor_bad} (must be 0)")
    print(f"PSNR(RGB) {psnr(mse_rgb):.2f} dB   PSNR(A) {psnr(mse_a):.2f} dB   PSNR(PM) {psnr(mse_pm):.2f} dB")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
