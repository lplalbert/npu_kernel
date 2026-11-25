#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 problemId 查我们每个题的最新提交（替代 sids 记录）。"""
import json, urllib.request, urllib.error

JWT = "<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json",
     "User-Agent": "Mozilla/5.0"}
BASE = "https://cannjudge.cn"

PROBS = [
    ("flashattention", "6acef0f7e76989292713697f"),
    ("convtranspose2d", "6acef213e769892927149a50"),
    ("gf2poly", "6aec3c8d1dac5d235b9a3c50"),
    ("gf2k_mul_square", "6aec3a341dac5d235b9509a4"),
    ("mhcsinkhorngrad", "6acef064e76989292712c8e3"),
    ("quantmatmulsigmoidmulquant", "6acef084e76989292712e8ef"),
    ("multiaddrmsnormdynamicquant", "6acef047e76989292712aaae"),
    ("confusionmulgrad", "6acef021e769892927127ff1"),
    ("bit_plane_transform", "6aec3d2e1dac5d235b9ba60d"),
    ("mseloss", "6ad9c4a23ff2fef7e81cf6c6"),
    ("group_norm", "6ad9c2a83ff2fef7e81bf865"),
    ("groupnormsiluquant", "6ad9c36f3ff2fef7e81c55cc"),
    ("index", "6ad9c4023ff2fef7e81ca27e"),
    ("trace", "6aceefd6e769892927123166"),
]

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

ids = ",".join(p[1] for p in PROBS)
st, body = get(f"/api/submissions/user/{USER}/problem-progress?problemIds={ids}")
print("status", st, "len", len(body))
try:
    d = json.loads(body)
except Exception:
    print(body[:600]); raise SystemExit
rows = d if isinstance(d, list) else (d.get("data") or d.get("rows") or [])
if isinstance(rows, dict):
    rows = [rows]
print("rows:", len(rows))
name_of = {p[1]: p[0] for p in PROBS}
for r in rows:
    pid = r.get("problem_id") or r.get("problemId")
    sid = r.get("submission_id") or r.get("_id") or r.get("latest_submission_id")
    stt = r.get("status") or r.get("submission_status")
    prec = r.get("precision_ratio")
    print(f"  {name_of.get(pid, pid):30s} sid={sid} status={stt} prec={prec}")
