# Verify bc6h_compress.hlsl's quantize10() picks the nearest 10-bit code for
# every non-negative finite half bit pattern, per the BC6H UF16 decode chain.

def unquantize(x, bits=10):
    if bits >= 15: return x
    if x == 0: return 0
    if x == (1 << bits) - 1: return 0xFFFF
    return ((x << 16) + 0x8000) >> bits

def decode(q):
    return (unquantize(q) * 31) >> 6

def quantize10_shader(h):
    q = (h * 2 + 1) // 62
    if q == 0 and h > 22: q = 1
    q = min(q, 1023)
    if q == 1023 and h < 31720: q = 1022
    return q

dec = [decode(q) for q in range(1024)]
assert dec[0] == 0 and dec[1023] == 0x7BFF, (dec[0], hex(dec[1023]))
# interior closed form used in the shader/doc
for q in range(1, 1023):
    assert dec[q] == 31 * q + 15, (q, dec[q])

worst = 0
bad = 0
for h in range(0x7BFF + 1):
    q = quantize10_shader(h)
    err = abs(dec[q] - h)
    best = min(abs(d - h) for d in (dec[max(q-1,0)], dec[q], dec[min(q+1,1023)]))
    if err != best:
        bad += 1
        if bad < 10: print(f"h={h}: shader q={q} err={err}, best neighbor err={best}")
    worst = max(worst, err)

print(f"codes checked: {0x7BFF+1}, non-nearest picks: {bad}, worst abs err: {worst}")
