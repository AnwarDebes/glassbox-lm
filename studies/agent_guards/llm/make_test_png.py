#!/usr/bin/env python
"""Write llm/test_shapes.png (standard library only): 256 x 256, white, a red filled circle on the left half and a blue
filled square on the right half. For the vision smoke test: the model should name a red circle and a blue square."""
import struct, zlib
from pathlib import Path
W = H = 256
rows = []
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        if (x - 64) ** 2 + (y - 128) ** 2 <= 48 ** 2:
            row += bytes((220, 20, 20))
        elif 150 <= x <= 230 and 88 <= y <= 168:
            row += bytes((20, 40, 220))
        else:
            row += bytes((255, 255, 255))
    rows.append(bytes(row))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
out = Path(__file__).resolve().parent / "test_shapes.png"
assert not out.exists()
out.write_bytes(png)
print(out, len(png))
