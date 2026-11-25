#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轮询已提交的 submissionId，打印 status / precision / 各 case。"""
import json, urllib.request, urllib.error, sys, time

JWT = "<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
BASE = "https://cannjudge.cn"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}

known = {}
for l in open("/tmp/ladder_sids.txt"):
    p = l.rstrip("\n").split("\t")
    if len(p) == 3:
        known[p[0]] = p[2]
known.setdefault("flashattention", "6b05c00985ad43bbe0653413")

def get(sid):
    try:
        r = urllib.request.Request(f"{BASE}/api/submissions/{sid}?userId={USER}", headers=H)
        with urllib.request.urlopen(r, timeout=40) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"_err": str(e)}

for name, sid in known.items():
    d = get(sid)
    if "_err" in d:
        print(f"{name:30s} ERR {d['_err'][:60]}"); continue
    st = d.get("status")
    res = d.get("result") or []
    vis = [x for x in res if x.get("testcase_status") not in (None, "Hidden")]
    passed = sum(1 for x in res if x.get("testcase_status") == "Pass")
    prec = [x.get("precision_ratio") for x in vis]
    times = [round(x.get("time") or 0, 2) for x in vis[:6]]
    print(f"{name:30s} {st:16s} pass={passed}/{len(res)} prec={prec[:6]} t={times}")
