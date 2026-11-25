#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验每个 zip 的 7 个成员是否与对应源码目录逐字节一致。"""
import os, zipfile, hashlib, glob

# name -> (zip, source_dir)
T = [
    ("trace",            "/root/work/ladder/trace_7files.zip",            "/root/work/ladder/trace"),
    ("marn_official",    "/root/work/ladder/marn_official_7files.zip",    "/root/work/ladder/marn_official"),
    ("cmg",              "/root/work/ladder/cmg_7files.zip",              "/root/work/ladder/cmg"),
    ("bpt",              "/root/work/ladder/bpt_7files.zip",              "/root/work/ladder/bpt"),
    ("gf2",              "/root/work/ladder/gf2_7files.zip",              "/root/work/ladder/gf2"),
    ("msg",              "/root/work/ladder/msg_7files.zip",              "/root/work/ladder/msg"),
    ("fa",               "/root/work/ladder/fa_7files.zip",               "/root/work/ladder/fa"),
    ("ct2d",             "/root/work/ladder/ct2d_7files.zip",             "/root/work/ladder/ct2d"),
    ("gf2poly",          "/root/work/ladder/gf2poly_7files.zip",          "/root/work/ladder/gf2poly"),
    ("qmsq",             "/root/work/ladder/qmsq_7files.zip",             "/root/work/ladder/qmsq"),
    ("mse",              "/root/work/ladder/mse_7files.zip",              "/root/work/ladder/mse"),
    ("gn",               "/root/work/ladder/gn_7files.zip",               "/root/work/ladder/gn"),
    ("gnsq",             "/root/work/ladder/gnsq_7files.zip",             "/root/work/ladder/gnsq"),
    ("index",            "/root/work/ladder/index_7files.zip",            "/root/work/ladder/index"),
]

def md5(b):
    return hashlib.md5(b).hexdigest()

bad_total = 0
for name, zp, src in T:
    if not os.path.exists(zp):
        print(f"{name:16s} ZIP-MISSING"); bad_total += 1; continue
    try:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            diffs = []
            for n in names:
                sp = os.path.join(src, n)
                if not os.path.exists(sp):
                    diffs.append(f"{n}:SRC-MISSING"); continue
                if md5(z.read(n)) != md5(open(sp, "rb").read()):
                    diffs.append(n)
            status = "OK" if (len(names) == 7 and not diffs) else "MISMATCH"
            if status != "OK": bad_total += 1
            print(f"{name:16s} {status:9s} entries={len(names)} diffs={diffs if diffs else '-'}")
    except Exception as e:
        print(f"{name:16s} ERR {e}"); bad_total += 1

print("\n=== 不一致/缺失总数:", bad_total, "===")
