#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每题(实现7文件 + NOTES + 题面摘要)汇总成 review/<name>.md，供 Codex 评审。"""
import os, zipfile, glob, json

OUT = "/root/work/ladder/review"
os.makedirs(OUT, exist_ok=True)

ZIPS = sorted(glob.glob("/root/work/ladder/*_7files.zip"))
SPEC_DIR = "/root/work/ladder_specs"
# zip 名 -> (题面文件名, NOTES 路径候选)
MAP = {
    "trace":            ("Trace.md",                        "/root/work/ladder/TRACE_NOTES.md"),
    "marn_official":    ("MultiAddRmsNormDynamicQuant.md",   "/root/work/ladder/marn_official/README_IMPL.md"),
    "cmg":              ("ConfusionMulGrad.md",              "/root/work/ladder/CMG_NOTES.md"),
    "bpt":              (None,                               "/root/work/ladder/BPT_NOTES.md"),
    "gf2":              (None,                               None),
    "msg":              (None,                               None),
    "fa":               ("FlashAttention.md",                "/root/work/ladder/FA_NOTES.md"),
    "ct2d":             ("ConvTranspose2D.md",               None),
    "gf2poly":          (None,                               None),
    "qmsq":             (None,                               "/root/work/ladder/QMSQ_NOTES.md"),
    "mse":              ("MseLoss.md",                       "/root/work/ladder/MSE_NOTES.md"),
    "gn":               ("GroupNorm.md",                     "/root/work/ladder/GN_NOTES.md"),
    "gnsq":             ("GroupNormSiluQuant.md",             "/root/work/ladder/GNSQ_NOTES.md"),
    "index":            ("Index.md",                         "/root/work/ladder/INDEX_NOTES.md"),
}

manifest = []
for zp in ZIPS:
    base = os.path.basename(zp).replace("_7files.zip", "")
    spec_name, notes_path = MAP.get(base, (None, None))
    parts = [f"# {base}\n"]
    if spec_name:
        sp = os.path.join(SPEC_DIR, spec_name)
        if os.path.exists(sp):
            parts.append("## SPEC\n```\n" + open(sp, encoding="utf-8").read()[:9000] + "\n```\n")
    if notes_path and os.path.exists(notes_path):
        parts.append("## NOTES\n```\n" + open(notes_path, encoding="utf-8").read()[:7000] + "\n```\n")
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            txt = z.read(n).decode("utf-8", "replace")
            parts.append(f"## FILE {n}\n```cpp\n{txt}\n```\n")
    dst = os.path.join(OUT, base + ".md")
    content = "\n".join(parts)
    open(dst, "w", encoding="utf-8").write(content)
    manifest.append((base, os.path.getsize(dst), len(content)))
    print(f"{base:20s} {os.path.getsize(dst):>8d} bytes")

print("\ntotal files:", len(manifest))
open(os.path.join(OUT, "_MANIFEST.md"), "w", encoding="utf-8").write(
    "# 评审包清单\n\n" + "\n".join(f"- {b}.md  ({s} bytes)" for b, s, _ in manifest))
