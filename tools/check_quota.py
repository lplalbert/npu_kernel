#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, urllib.request
JWT = "<CANNJUDGE_JWT_REDACTED>.<CANNJUDGE_JWT_REDACTED>"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
CID = "6aced8d2e769892927fdb17f"

r = urllib.request.Request(f"https://cannjudge.cn/api/contests/{CID}/my-submission-quota", headers=H)
print("quota:", urllib.request.urlopen(r, timeout=30).read().decode())

# 列出已提交（从 sids 文件）
import os
p = "/tmp/ladder_sids.txt"
if os.path.exists(p):
    lines = [l.strip() for l in open(p) if l.strip()]
    print("submitted:", len(lines))
    for l in lines:
        print("  ", l)
else:
    print("no sids file")
