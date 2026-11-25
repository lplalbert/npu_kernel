#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取本账号全部 38 道题的排名/成绩，产出 handoff_scores.json。"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probids import LADDER, MFG, S10

JWT = os.environ.get("CANNJUDGE_JWT", "")
USER = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
BASE = "https://cannjudge.cn"


def get(path):
    try:
        r = urllib.request.Request(BASE + path, headers=H)
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:150]
    except Exception as e:
        return -1, str(e)


def find_mine(pid):
    for page in range(1, 12):
        st, body = get(f"/api/problems/{pid}/ranking?page={page}&size=100")
        if st != 200:
            return None, f"HTTP {st}"
        try:
            d = json.loads(body)
        except Exception:
            return None, "non-json"
        for r in (d.get("rows") or []):
            if str(r.get("user_id")) == USER:
                return r, None
        if page >= int(d.get("pages") or 1):
            break
        time.sleep(0.25)
    return None, "not ranked"


def run(kind, items):
    out = []
    for it in items:
        name, pid = it[0], it[1]
        r, err = find_mine(pid)
        rec = {"kind": kind, "name": name, "problemId": pid}
        if r:
            sub = r.get("submission") or {}
            res = r.get("result") or []
            precs = [x.get("precision_ratio") for x in res if x.get("precision_ratio") is not None]
            rec.update({
                "rank": r.get("rank"),
                "score": round(float(r.get("score") or 0), 2),
                "submissionId": sub.get("_id") or r.get("submission_id"),
                "status": sub.get("status") or r.get("status"),
                "cases": len(res),
                "n_bad": sum(1 for x in res if x.get("testcase_status") != "Pass"),
                "worst_precision": min(precs) if precs else None,
                "best_time": round(min([x.get("time") or 1e18 for x in res] or [0]), 2),
            })
        else:
            rec["error"] = err
        out.append(rec)
        print(f"{kind:6s} {name:24s} rank={rec.get('rank')} score={rec.get('score')} "
              f"status={rec.get('status')} bad={rec.get('n_bad')} err={rec.get('error','')}")
    return out


allrecs = run("ladder", LADDER) + run("mfg", MFG) + run("s10", S10)
with open("handoff_scores.json", "w", encoding="utf-8") as f:
    json.dump(allrecs, f, ensure_ascii=False, indent=2)
print(f"\nwritten handoff_scores.json  ({len(allrecs)} problems)")
