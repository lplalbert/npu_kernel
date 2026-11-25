#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CANNJudge 拉取本账号在各题上的最新提交（submissionId/status/precision）。"""
import json
import os
import sys
import urllib.error
import urllib.request

JWT = os.environ.get("CANNJUDGE_JWT", "")
USER = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
H = {"Cookie": f"cannjudge_auth={JWT}", "Accept": "application/json",
     "User-Agent": "Mozilla/5.0"}
BASE = "https://cannjudge.cn"

LADDER = {
    "trace": "6aceefd6e769892927123166",
    "multi_add_rms_norm_dynamic_quant": "6acef047e76989292712aaae",
    "confusion_mul_grad": "6acef021e769892927127ff1",
    "bit_plane_transform": "6aec3d2e1dac5d235b9ba60d",
    "gf2k_mul_square": "6aec3a341dac5d235b9509a4",
    "gf2_poly": "6aec3c8d1dac5d235b9a3c50",
    "mhc_sinkhorn_grad": "6acef064e76989292712c8e3",
    "flash_attention": "6acef0f7e76989292713697f",
    "conv_transpose2_d": "6acef213e769892927149a50",
    "quant_matmul_sigmoid_mul_quant": "6acef084e76989292712e8ef",
    "mse_loss": "6ad9c4a23ff2fef7e81cf6c6",
    "group_norm": "6ad9c2a83ff2fef7e81bf865",
    "group_norm_silu_quant": "6ad9c36f3ff2fef7e81c55cc",
    "index": "6ad9c4023ff2fef7e81ca27e",
    "bn_training_update": "6ad9c0583ff2fef7e81ab6bb",
    "depthwise_conv2_d": "6ad9c1ff3ff2fef7e81b9d0e",
    "cube1": "6ad9c71b3ff2fef7e81e2f75",
    "cube2": "6ad9c7f83ff2fef7e81e894c",
}
MFG = {
    "spspmm": "6af276736bd0f9be37a4cb08",
    "fourier_transform2d": "6af2d6646bd0f9be375ab041",
    "fused_prop_step": "6af2d5b76bd0f9be3759446c",
}


def get(path):
    try:
        r = urllib.request.Request(BASE + path, headers=H)
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        return -1, str(e)


def ranking_latest(pid):
    """从 ranking 接口找本账号最新一条（含 submissionId / score / best_time）。"""
    st, body = get(f"/api/problems/{pid}/ranking?page=1&size=100")
    if st != 200:
        return None, f"ranking {st}: {body[:120]}"
    try:
        d = json.loads(body)
    except Exception:
        return None, "ranking non-json"
    rows = d if isinstance(d, list) else (d.get("data") or d.get("list") or d.get("rows") or [])
    if isinstance(rows, dict):
        rows = rows.get("list") or rows.get("data") or []
    mine = [r for r in rows if str(r.get("user_id") or r.get("userId") or "") == USER]
    return (mine or rows), None


if __name__ == "__main__":
    print("尝试 ranking 接口…")
    st, body = get("/api/problems/6aceefd6e769892927123166/ranking?page=1&size=5")
    print("status:", st)
    print(body[:500])
