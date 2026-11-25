#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每个 zip 的 7 个文件里的非 ASCII 字符清掉（只影响注释/错误串），
同时写回源目录，保证 zip == source。然后重新打包。"""
import os, re, zipfile

SETS = {
    "trace":            ("/root/work/ladder/trace",            "trace",                          "/root/work/ladder/trace_7files.zip"),
    "marn_official":    ("/root/work/ladder/marn_official",    "multi_add_rms_norm_dynamic_quant","/root/work/ladder/marn_official_7files.zip"),
    "cmg":              ("/root/work/ladder/cmg",              "confusion_mul_grad",             "/root/work/ladder/cmg_7files.zip"),
    "bpt":              ("/root/work/ladder/bpt",              "bit_plane_transform",            "/root/work/ladder/bpt_7files.zip"),
    "gf2":              ("/root/work/ladder/gf2",              "gf2k_mul_square",                "/root/work/ladder/gf2_7files.zip"),
    "msg":              ("/root/work/ladder/msg",              "mhc_sinkhorn_grad",              "/root/work/ladder/msg_7files.zip"),
    "fa":               ("/root/work/ladder/fa",               "flash_attention",                "/root/work/ladder/fa_7files.zip"),
    "ct2d":             ("/root/work/ladder/ct2d",             "conv_transpose2_d",              "/root/work/ladder/ct2d_7files.zip"),
    "gf2poly":          ("/root/work/ladder/gf2poly",          "gf2_poly",                       "/root/work/ladder/gf2poly_7files.zip"),
    "qmsq":             ("/root/work/ladder/qmsq",             "quant_matmul_sigmoid_mul_quant", "/root/work/ladder/qmsq_7files.zip"),
    "mse":              ("/root/work/ladder/mse",              "mse_loss",                       "/root/work/ladder/mse_7files.zip"),
    "gn":               ("/root/work/ladder/gn",               "group_norm",                     "/root/work/ladder/gn_7files.zip"),
    "gnsq":             ("/root/work/ladder/gnsq",             "group_norm_silu_quant",          "/root/work/ladder/gnsq_7files.zip"),
    "index":            ("/root/work/ladder/index",            "index",                          "/root/work/ladder/index_7files.zip"),
}

def strip_nonascii(txt):
    out = []
    for ch in txt:
        out.append(ch if ord(ch) < 128 else "")
    s = "".join(out)
    # 收拾可能出现的空注释行与多余空格
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

for name, (src, base, zp) in SETS.items():
    pairs = [("op_host", f"{base}_def.cpp"),
             ("op_host", f"{base}_infershape.cpp"),
             ("op_host", f"{base}_tiling.cpp"),
             ("op_kernel", f"{base}.cpp"),
             ("op_kernel", f"{base}.h"),
             ("op_kernel", f"{base}_tiling_data.h"),
             ("op_kernel", f"{base}_tiling_key.h")]
    missing = [f"{d}/{f}" for d, f in pairs if not os.path.exists(os.path.join(src, d, f))]
    if missing:
        print(f"{name}: SKIP missing {missing}"); continue
    changed = []
    for d, f in pairs:
        p = os.path.join(src, d, f)
        txt = open(p, encoding="utf-8", errors="replace").read()
        new = strip_nonascii(txt)
        if new != txt:
            open(p, "w", encoding="utf-8").write(new)
            changed.append(f"{d}/{f}")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for d, f in pairs:
            z.write(os.path.join(src, d, f), arcname=f"{d}/{f}")
    with zipfile.ZipFile(zp) as z:
        # 复查非 ASCII
        left = 0
        for n in z.namelist():
            if n.endswith("/"): continue
            t = z.read(n).decode("utf-8", "replace")
            left += sum(1 for c in t if ord(c) > 127)
    print(f"{name:16s} cleaned={len(changed):2d} files, nonascii_left={left}, entries=7")
print("done")
