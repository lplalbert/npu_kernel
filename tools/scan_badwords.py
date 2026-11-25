#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描全部 zip 的 cout/cin/cerr/printf/getenv 等标识符（判题机禁词），并可选修复。"""
import os, re, zipfile, sys

SETS = {
    "trace":         ("/root/work/ladder/trace",         "trace",                          "/root/work/ladder/trace_7files.zip"),
    "marn_official": ("/root/work/ladder/marn_official", "multi_add_rms_norm_dynamic_quant","/root/work/ladder/marn_official_7files.zip"),
    "cmg":           ("/root/work/ladder/cmg",           "confusion_mul_grad",             "/root/work/ladder/cmg_7files.zip"),
    "bpt":           ("/root/work/ladder/bpt",           "bit_plane_transform",            "/root/work/ladder/bpt_7files.zip"),
    "gf2":           ("/root/work/ladder/gf2",           "gf2k_mul_square",                "/root/work/ladder/gf2_7files.zip"),
    "msg":           ("/root/work/ladder/msg",           "mhc_sinkhorn_grad",              "/root/work/ladder/msg_7files.zip"),
    "fa":            ("/root/work/ladder/fa",            "flash_attention",                "/root/work/ladder/fa_7files.zip"),
    "ct2d":          ("/root/work/ladder/ct2d",          "conv_transpose2_d",              "/root/work/ladder/ct2d_7files.zip"),
    "gf2poly":       ("/root/work/ladder/gf2poly",       "gf2_poly",                       "/root/work/ladder/gf2poly_7files.zip"),
    "qmsq":          ("/root/work/ladder/qmsq",          "quant_matmul_sigmoid_mul_quant", "/root/work/ladder/qmsq_7files.zip"),
    "mse":           ("/root/work/ladder/mse",           "mse_loss",                       "/root/work/ladder/mse_7files.zip"),
    "gn":            ("/root/work/ladder/gn",            "group_norm",                     "/root/work/ladder/gn_7files.zip"),
    "gnsq":          ("/root/work/ladder/gnsq",          "group_norm_silu_quant",          "/root/work/ladder/gnsq_7files.zip"),
    "index":         ("/root/work/ladder/index",         "index",                          "/root/work/ladder/index_7files.zip"),
}

# 判题机禁词（标识符级）
BAD = re.compile(r'\b(cout|cin|cerr|clog|printf|fprintf|sprintf|snprintf|getenv|system|popen)\b')
FIX = {"cout": "chOut", "cin": "chIn", "cerr": "chErr", "clog": "chLog"}

do_fix = "--fix" in sys.argv

def pairs_of(base):
    return [("op_host", f"{base}_def.cpp"),
            ("op_host", f"{base}_infershape.cpp"),
            ("op_host", f"{base}_tiling.cpp"),
            ("op_kernel", f"{base}.cpp"),
            ("op_kernel", f"{base}.h"),
            ("op_kernel", f"{base}_tiling_data.h"),
            ("op_kernel", f"{base}_tiling_key.h")]

bad = 0
for name, (src, base, zp) in SETS.items():
    pairs = pairs_of(base)
    if any(not os.path.exists(os.path.join(src, d, f)) for d, f in pairs):
        print(f"{name:16s} SKIP (missing src)"); continue
    hits = {}
    for d, f in pairs:
        p = os.path.join(src, d, f)
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in BAD.finditer(txt):
            hits.setdefault(m.group(1), []).append(f"{d}/{f}")
    if not hits:
        print(f"{name:16s} OK")
        continue
    bad += 1
    print(f"{name:16s} BAD: " + ", ".join(f"{k}×{len(v)}" for k, v in hits.items()))
    if do_fix:
        for d, f in pairs:
            p = os.path.join(src, d, f)
            txt = open(p, encoding="utf-8", errors="replace").read()
            new = BAD.sub(lambda m: FIX.get(m.group(1), m.group(1)), txt)
            if new != txt:
                open(p, "w", encoding="utf-8").write(new)
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for d, f in pairs:
                z.write(os.path.join(src, d, f), arcname=f"{d}/{f}")
        # 复查
        left = 0
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith("/"): continue
                left += len(BAD.findall(z.read(n).decode("utf-8", "replace")))
        print(f"{'':16s} -> fixed & repacked, residual_bad={left}")
print(f"\n=== 有禁词的文件数: {bad}/{len(SETS)} ===")
