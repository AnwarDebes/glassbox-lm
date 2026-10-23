#!/usr/bin/env python
"""Convert routerbench_0shot.pkl (withmartian/routerbench) to parquet with a restricted unpickler
that only resolves the pandas/numpy reconstruction functions the file references (checked first
with pickletools: DataFrame, BlockManager, _unpickle_block, _reconstruct, ndarray, dtype, slice,
_frombuffer, _new_Index, Index, RangeIndex). Anything else raises."""
import pickle
import sys

ALLOWED = {
    ("pandas.core.frame", "DataFrame"), ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
    ("builtins", "slice"), ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    df = Restricted(f).load()
print(type(df), df.shape)
print(list(df.columns))
df.columns = [str(c) for c in df.columns]
df.to_parquet(dst)
print("wrote", dst)
