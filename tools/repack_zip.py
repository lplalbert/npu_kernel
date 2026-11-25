#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按源码目录重新打包指定算子的 7 文件 zip（无前缀）。"""
import os, sys, zipfile

SPEC = {
    "ct2d": ("/root/work/ladder/ct2d", "conv_transpose2_d"),
    "trace": ("/root/work/ladder/trace", "trace"),
}

def repack(name):
    src, base = SPEC[name]
    out = f"/root/work/ladder/{'ct2d' if name=='ct2d' else 'trace'}_7files.zip"
    pairs = [("op_host", f"{base}_def.cpp"),
             ("op_host", f"{base}_infershape.cpp"),
             ("op_host", f"{base}_tiling.cpp"),
             ("op_kernel", f"{base}.cpp"),
             ("op_kernel", f"{base}.h"),
             ("op_kernel", f"{base}_tiling_data.h"),
             ("op_kernel", f"{base}_tiling_key.h")]
    missing = [f"{d}/{f}" for d, f in pairs if not os.path.exists(os.path.join(src, d, f))]
    # ct2d 的 def 文件名是 conv_transpose2_d_def.cpp（含 2）
    if missing:
        print(name, "candidates missing:", missing)
        # 实际列一下目录
        for d in ("op_host", "op_kernel"):
            print(f"  {d}:", sorted(os.listdir(os.path.join(src, d))))
        return False
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, f in pairs:
            z.write(os.path.join(src, d, f), arcname=f"{d}/{f}")
    with zipfile.ZipFile(out) as z:
        ns = z.namelist()
    print(f"{name}: repacked {out} entries={len(ns)}")
    return True

for n in sys.argv[1:] or ["ct2d"]:
    repack(n)
