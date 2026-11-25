#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取本账号全部 38 道题的排名/成绩，产出 handoff_scores.json。"""
import json
import time
import urllib.error
import urllib.request

JWT = ("<CANNJUDGE_JWT_REDACTED>"
       ".<CANNJUDGE_JWT_REDACTED>")
USER = "<MY_USER_ID>"
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
BASE = "https://cannjudge.cn"

LADDER = [
    ("trace", "6aceefd6e769892927123166", 1),
    ("multi_add_rms_norm_dynamic_quant", "6acef047e76989292712aaae", 2),
    ("confusion_mul_grad", "6acef021e769892927127ff1", 2),
    ("bit_plane_transform", "6aec3d2e1dac5d235b9ba60d", 2),
    ("gf2k_mul_square", "6aec3a341dac5d235b9509a4", 3),
    ("gf2_poly", "6aec3c8d1dac5d235b9a3c50", 4),
    ("mhc_sinkhorn_grad", "6acef064e76989292712c8e3", 3),
    ("flash_attention", "6acef0f7e76989292713697f", 5),
    ("conv_transpose2_d", "6acef213e769892927149a50", 5),
    ("quant_matmul_sigmoid_mul_quant", "6acef084e76989292712e8ef", 3),
    ("mse_loss", "6ad9c4a23ff2fef7e81cf6c6", 2),
    ("group_norm", "6ad9c2a83ff2fef7e81bf865", 2),
    ("group_norm_silu_quant", "6ad9c36f3ff2fef7e81c55cc", 2),
    ("index", "6ad9c4023ff2fef7e81ca27e", 2),
    ("bn_training_update", "6ad9c0583ff2fef7e81ab6bb", 1),
    ("depthwise_conv2_d", "6ad9c1ff3ff2fef7e81b9d0e", 5),
    ("cube1", "6ad9c71b3ff2fef7e81e2f75", 4),
    ("cube2", "6ad9c7f83ff2fef7e81e894c", 4),
]
MFG = [
    ("cylinderquery", "6af1c67a6bd0f9be3792fb4a"), ("furthestpointsample", "6af1c8736bd0f9be37971169"),
    ("ballquery", "6af1c92b6bd0f9be3798a712"), ("radius", "6af1ca066bd0f9be379a6c84"),
    ("gridcluster", "6af1cab46bd0f9be379bc6c6"), ("gathercoo", "6af1cbe56bd0f9be379e1c23"),
    ("segmentmincsr", "6af1cdaa6bd0f9be37a1afe1"), ("spmmsum", "6af275c16bd0f9be37a3bbd5"),
    ("spspmm", "6af276736bd0f9be37a4cb08"), ("randomwalk", "6af277846bd0f9be37a6755a"),
    ("complexabs", "6af2d3706bd0f9be375466ed"), ("complexdiv", "6af2d43a6bd0f9be3756251e"),
    ("fused_prop_step", "6af2d5b76bd0f9be3759446c"), ("fourier_transform2d", "6af2d6646bd0f9be375ab041"),
    ("complexexp", "6af6cbd36bd0f9be37ba0aae"),
]
S10 = [
    ("mish", "6ad9cfab3ff2fef7e8227643"), ("bessel_i0e", "6ad9d3523ff2fef7e82428fe"),
    ("dot", "6ad9d4bb3ff2fef7e824d520"), ("poisson_nll_loss", "6ad9d60b3ff2fef7e8257053"),
    ("reduce_log_sum_exp", "6ad9d7423ff2fef7e82610ca"),
]


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
