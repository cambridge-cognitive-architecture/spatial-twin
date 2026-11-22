"""Memory-mapped reader for binary little-endian, vertex-only PLY."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TYPES = dict(char="i1", int8="i1", uchar="u1", uint8="u1", short="<i2", int16="<i2",
             ushort="<u2", uint16="<u2", int="<i4", int32="<i4", uint="<u4", uint32="<u4",
             float="<f4", float32="<f4", double="<f8", float64="<f8")


def load(path: str | Path, stride: int = 1) -> tuple[np.ndarray, int]:
    path = Path(path)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    with path.open("rb") as fh:
        raw = fh.read(65536)
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("no end_header in the first 64 kB")
    text = raw[:end].decode("ascii", errors="replace")
    if not re.search(r"^format\s+binary_little_endian", text, re.M):
        raise ValueError("only binary_little_endian is supported")
    elements = re.findall(r"^element\s+(\S+)\s+(\d+)", text, re.M)
    if len(elements) != 1 or re.search(r"^property\s+list", text, re.M):
        raise ValueError("only a single element without list properties is supported")
    props = re.findall(r"^property\s+(\S+)\s+(\S+)", text, re.M)
    dtype = np.dtype([(name, TYPES[kind]) for kind, name in props])
    count = int(elements[0][1])
    offset = raw.find(b"\n", end) + 1
    if path.stat().st_size < offset + count * dtype.itemsize:
        raise ValueError("file is truncated")
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    return data[::stride], count
