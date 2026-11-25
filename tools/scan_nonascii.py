#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描全部 zip 的非 ASCII（中文）残留 —— 这是判题机"不合规内容"的触发点。"""
import os, zipfile, glob

zips = sorted(glob.glob("/root/work/ladder/*_7files.zip"))
total = 0
for zp in zips:
    name = os.path.basename(zp).replace("_7files.zip", "")
    hits = []
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            txt = z.read(n).decode("utf-8", "replace")
            for i, l in enumerate(txt.splitlines(), 1):
                bad = [c for c in l if ord(c) > 127]
                if bad:
                    hits.append((n, i, l.strip()[:100]))
    if hits:
        total += 1
        print(f"### {name}  ({len(hits)} 行含非ASCII)")
        for h in hits[:12]:
            print(f"    {h[0]}:{h[1]}  {h[2]}")
        if len(hits) > 12:
            print(f"    ... 共 {len(hits)} 行")
    else:
        print(f"OK  {name}")
print(f"\n=== 含非ASCII的文件数: {total}/{len(zips)} ===")
