#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用指定目录的 7 文件提交，诊断"不合规内容"到底来自哪。
用法: python3 try_submit.py <src_dir> <base> <problemId> <tag>
"""
import os, sys, json, zipfile, time, urllib.request, urllib.error

JWT = "<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
BASE = "https://cannjudge.cn"
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}

src, base, pid, tag = sys.argv[1:5]
pairs = [("op_host", f"{base}_def.cpp"),
         ("op_host", f"{base}_infershape.cpp"),
         ("op_host", f"{base}_tiling.cpp"),
         ("op_kernel", f"{base}.cpp"),
         ("op_kernel", f"{base}.h"),
         ("op_kernel", f"{base}_tiling_data.h"),
         ("op_kernel", f"{base}_tiling_key.h")]
files = []
for d, f in pairs:
    p = os.path.join(src, d, f)
    if not os.path.exists(p):
        print("MISSING", p); sys.exit(1)
    files.append({"path": f"{d}/{f}", "content": open(p, encoding="utf-8", errors="replace").read()})
print(f"[{tag}] files={len(files)} total_chars={sum(len(f['content']) for f in files)}")
body = json.dumps({"problemId": pid, "userId": USER, "files": files}).encode()
r = urllib.request.Request(BASE + "/api/submissions/submit", headers=H, data=body, method="POST")
try:
    with urllib.request.urlopen(r, timeout=180) as resp:
        print(f"[{tag}] HTTP {resp.status} {resp.read().decode('utf-8','replace')[:300]}")
except urllib.error.HTTPError as e:
    print(f"[{tag}] HTTP-ERR {e.code} {e.read().decode('utf-8','replace')[:300]}")
except Exception as e:
    print(f"[{tag}] EXC {type(e).__name__} {e}")
