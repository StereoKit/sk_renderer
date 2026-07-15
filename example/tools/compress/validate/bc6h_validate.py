# BC6H mode-11 CPU reference decoder + PSNR vs the exact encoder input.
#
# Reference chain reproduces what the GPU encoder saw: Radiance RGBE file
# -> rg11b10 (hdr_load.h integer conversion, replicated exactly) -> float
# -> half bits (exact: uf11/uf10 mantissas fit in half). Decode chain is
# implemented straight from the D3D spec, independent of the shader.
import struct, sys
import numpy as np

def load_radiance(path):
    with open(path, 'rb') as f:
        data = f.read()
    # header
    pos = data.index(b'\n\n') if b'\n\n' in data[:200] else None
    lines_end = 0
    i = 0
    while True:
        j = data.index(b'\n', i)
        line = data[i:j]
        i = j + 1
        if line == b'':
            break
    # resolution line
    j = data.index(b'\n', i)
    res = data[i:j].split()
    assert res[0] == b'-Y' and res[2] == b'+X', res
    h, w = int(res[1]), int(res[3])
    i = j + 1
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        if w < 8 or w > 32767 or data[i] != 2 or data[i+1] != 2 or ((data[i+2] << 8) | data[i+3]) != w:
            # old-style flat scanline
            row = np.frombuffer(data[i:i + 4*w], dtype=np.uint8).reshape(w, 4)
            img[y] = row
            i += 4 * w
            continue
        i += 4
        for c in range(4):
            x = 0
            while x < w:
                count = data[i]; i += 1
                if count > 128:
                    img[y, x:x+count-128, c] = data[i]; i += 1
                    x += count - 128
                else:
                    img[y, x:x+count, c] = np.frombuffer(data[i:i+count], dtype=np.uint8)
                    i += count
                    x += count
    return img  # RGBE uint8

def clz8(x):
    n = 0
    if x == 0: return 8
    while not (x & 0x80):
        n += 1; x = (x << 1) & 0xFF
    return n

def rgbe_to_uf(m, e, man_bits):
    # replicates _rgbe_to_uf11/_rgbe_to_uf10 from hdr_load.h
    if m == 0: return 0
    msb = 7 - clz8(m)
    exp = msb + e - 121
    if exp >= 31:
        return (30 << man_bits) | ((1 << man_bits) - 1)
    if exp <= 0:
        lim = -5 if man_bits == 6 else -4
        if exp < lim: return 0
        shift = 7 - man_bits - exp
        if shift >= 8: return 0
        return m >> shift
    frac = m ^ (1 << msb)
    man = (frac >> (msb - man_bits)) if msb >= man_bits else (frac << (man_bits - msb))
    return (exp << man_bits) | (man & ((1 << man_bits) - 1))

def uf_to_float(v, man_bits):
    exp = v >> man_bits
    man = v & ((1 << man_bits) - 1)
    if exp == 0:
        return (man / (1 << man_bits)) * 2.0**-14
    return (1.0 + man / (1 << man_bits)) * 2.0**(exp - 15)

# BC6H UF16 decode (D3D spec)
W4 = [0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64]

def unquantize10(x):
    if x == 0: return 0
    if x == 1023: return 0xFFFF
    return ((x << 16) + 0x8000) >> 10

def half_to_float(h):
    return np.frombuffer(np.array([h], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)

class BitReader:
    def __init__(self, block16):
        self.v = int.from_bytes(block16, 'little')
        self.pos = 0
    def get(self, n):
        r = (self.v >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return r

def decode_block_mode11(block16):
    br = BitReader(block16)
    mode = br.get(5)
    assert mode == 0x03, f"not mode 11: mode bits {mode:#x}"
    e0 = [br.get(10), br.get(10), br.get(10)]
    e1 = [br.get(10), br.get(10), br.get(10)]
    idx = [br.get(3)]  # anchor: MSB implicit 0
    for _ in range(15):
        idx.append(br.get(4))
    u0 = [unquantize10(c) for c in e0]
    u1 = [unquantize10(c) for c in e1]
    out = np.zeros((16, 3), dtype=np.uint16)
    for p in range(16):
        w = W4[idx[p]]
        for c in range(3):
            interp = (u0[c] * (64 - w) + u1[c] * w + 32) >> 6
            out[p, c] = (interp * 31) >> 6
    return out, idx

def main(hdr_path, dds_path):
    rgbe = load_radiance(hdr_path)
    h, w = rgbe.shape[:2]

    # exact encoder input, in half bits
    src_half = np.zeros((h, w, 3), dtype=np.uint16)
    f16 = np.float16
    cache = {}
    for y in range(h):
        for x in range(w):
            r, g, b, e = rgbe[y, x]
            key = (r, g, b, e)
            if key not in cache:
                if e == 0:
                    vals = (0, 0, 0)
                else:
                    fr = uf_to_float(rgbe_to_uf(int(r), int(e), 6), 6)
                    fg = uf_to_float(rgbe_to_uf(int(g), int(e), 6), 6)
                    fb = uf_to_float(rgbe_to_uf(int(b), int(e), 5), 5)
                    vals = tuple(np.array([fr, fg, fb], dtype=f16).view(np.uint16))
                cache[key] = vals
            src_half[y, x] = cache[key]

    with open(dds_path, 'rb') as f:
        dds = f.read()
    assert dds[:4] == b'DDS ' and struct.unpack_from('<I', dds, 84)[0] == 0x30315844
    dh, dw = struct.unpack_from('<II', dds, 12)
    dxgi = struct.unpack_from('<I', dds, 128)[0]
    assert (dh, dw) == (h, w) and dxgi == 95, (dh, dw, dxgi)
    blocks = dds[148:]

    bw, bh = (w + 3) // 4, (h + 3) // 4
    dec_half = np.zeros((h, w, 3), dtype=np.uint16)
    anchor_msb_set = 0
    worst_block = (0, None)
    for by in range(bh):
        for bx in range(bw):
            blk = blocks[(by * bw + bx) * 16:(by * bw + bx) * 16 + 16]
            texels, idx = decode_block_mode11(blk)
            if idx[0] >= 8: anchor_msb_set += 1
            berr = 0
            for p in range(16):
                x = min(bx * 4 + p % 4, w - 1)
                y = min(by * 4 + p // 4, h - 1)
                if bx * 4 + p % 4 < w and by * 4 + p // 4 < h:
                    dec_half[y, x] = texels[p]
                d = texels[p].astype(np.int64) - src_half[min(by*4+p//4, h-1), min(bx*4+p%4, w-1)].astype(np.int64)
                berr = max(berr, int(np.abs(d).max()))
            if berr > worst_block[0]:
                worst_block = (berr, (bx, by))

    diff = dec_half.astype(np.int64) - src_half.astype(np.int64)
    mse = np.mean(diff.astype(np.float64) ** 2)
    psnr_half = 10 * np.log10(31743.0 ** 2 / mse) if mse > 0 else 999

    # LDR-comparable number: clamp to [0,1], srgb-encode, 8-bit, PSNR
    def to_ldr(half_img):
        f = half_img.view(np.float16).astype(np.float64)
        f = np.clip(f, 0, 1)
        s = np.where(f <= 0.0031308, f * 12.92, 1.055 * f ** (1 / 2.4) - 0.055)
        return np.round(s * 255)
    dl, sl = to_ldr(dec_half), to_ldr(src_half)
    mse_ldr = np.mean((dl - sl) ** 2)
    psnr_ldr = 10 * np.log10(255.0 ** 2 / mse_ldr) if mse_ldr > 0 else 999

    # linear relative error
    fd = dec_half.view(np.float16).astype(np.float64)
    fs = src_half.view(np.float16).astype(np.float64)
    rel = np.abs(fd - fs) / np.maximum(fs, 2.0**-6)
    print(f"{w}x{h}, {bw*bh} blocks, all mode 11")
    print(f"anchor MSB violations: {anchor_msb_set} (must be 0)")
    print(f"PSNR (half-bit space, encoder metric): {psnr_half:.2f} dB")
    print(f"PSNR (clamped-LDR sRGB 8-bit):          {psnr_ldr:.2f} dB")
    print(f"relative linear error: mean {rel.mean()*100:.3f}%  p99 {np.percentile(rel,99)*100:.3f}%  max {rel.max()*100:.2f}%")
    print(f"worst block: max {worst_block[0]} half-steps at {worst_block[1]}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
