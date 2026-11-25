#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印某次提交每 case 的精确 precision_ratio（repr 全精度）与 time。"""
import json, sys, urllib.request
JWT = "<CANNJUDGE_JWT_REDACTED>.<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
sid = sys.argv[1]
r = urllib.request.Request(f"https://cannjudge.cn/api/submissions/{sid}?userId={USER}", headers=H)
d = json.loads(urllib.request.urlopen(r, timeout=40).read().decode())
print("sid:", sid, "status:", d.get("status"))
print(f"{'case':>5s}  {'status':<14s} {'precision_ratio':<22s} {'time':<16s} best_time")
for i, x in enumerate(d.get("result") or []):
    p = x.get("precision_ratio")
    t = x.get("time")
    bt = x.get("best_time")
    ps = repr(p) if p is not None else "-"
    print(f"{i:>5d}  {str(x.get('testcase_status')):<14s} {ps:<22s} {str(t):<16s} {bt}")
