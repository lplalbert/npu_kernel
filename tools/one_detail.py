#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印某次提交的完整 result（含每个 case 的 msg / blockdim 等字段）。"""
import json, sys, urllib.request
JWT = "<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
sid = sys.argv[1]
r = urllib.request.Request(f"https://cannjudge.cn/api/submissions/{sid}?userId={USER}", headers=H)
d = json.loads(urllib.request.urlopen(r, timeout=40).read().decode())
print("status:", d.get("status"))
print("top keys:", [k for k in d.keys()])
res = d.get("result") or []
print("cases:", len(res))
for i, x in enumerate(res):
    print(f"\n--- case{i} ---")
    for k, v in x.items():
        s = str(v)
        if k == "msg":
            print(f"  msg: {s}")
            continue
        if len(s) > 300: s = s[:300] + "..."
        print(f"  {k}: {s}")
