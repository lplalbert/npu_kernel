#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, urllib.request, urllib.error

JWT = "<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
CID = "6aced8d2e769892927fdb17f"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
BASE = "https://cannjudge.cn"

def get(path):
    try:
        r = urllib.request.Request(BASE + path, headers=H)
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try: b = e.read().decode("utf-8", "replace")
        except Exception: b = ""
        return e.code, b
    except Exception as e:
        return -1, str(e)

print("== quota ==")
st, b = get(f"/api/contests/{CID}/my-submission-quota")
print(st, b[:200])

print("\n== global list (userId) ==")
for q in [f"/api/submissions/global/list?userId={USER}&page=1&size=20",
          f"/api/submissions/global/list?userId={USER}&page=1&per_page=20",
          f"/api/submissions/global/list?user_id={USER}&page=1&size=20"]:
    st, b = get(q)
    print(f"{st} {q[:60]} -> {b[:300]}")
    if st == 200:
        break

print("\n== contest submissions ==")
for q in [f"/api/submissions/contest/{CID}/my?page=1&size=20",
          f"/api/submissions/contest/{CID}/list?userId={USER}&page=1&size=20"]:
    st, b = get(q)
    print(f"{st} {q[:60]} -> {b[:300]}")
