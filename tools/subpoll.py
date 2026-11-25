#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用：提交指定题 → 遵守 429 cooldown → 轮询到出结果 → 打印逐 case 详情。
用法:  python3 subpoll.py <name> [<name> ...]
       python3 subpoll.py --detail <submissionId>     # 只查已有提交的详情
"""
import os, sys, json, time, zipfile, urllib.request, urllib.error

JWT = "<CANNJUDGE_JWT_REDACTED>.<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
BASE = "https://cannjudge.cn"
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
SIDS = "/tmp/ladder_sids.txt"

MAP = {
    "flashattention": ("6acef0f7e76989292713697f", "/root/work/ladder/fa_7files.zip"),
    "convtranspose2d": ("6acef213e769892927149a50", "/root/work/ladder/ct2d_7files.zip"),
    "gf2poly": ("6aec3c8d1dac5d235b9a3c50", "/root/work/ladder/gf2poly_7files.zip"),
    "gf2k_mul_square": ("6aec3a341dac5d235b9509a4", "/root/work/ladder/gf2_7files.zip"),
    "mhcsinkhorngrad": ("6acef064e76989292712c8e3", "/root/work/ladder/msg_7files.zip"),
    "quantmatmulsigmoidmulquant": ("6acef084e76989292712e8ef", "/root/work/ladder/qmsq_7files.zip"),
    "multiaddrmsnormdynamicquant": ("6acef047e76989292712aaae", "/root/work/ladder/marn_official_7files.zip"),
    "confusionmulgrad": ("6acef021e769892927127ff1", "/root/work/ladder/cmg_7files.zip"),
    "bit_plane_transform": ("6aec3d2e1dac5d235b9ba60d", "/root/work/ladder/bpt_7files.zip"),
    "mseloss": ("6ad9c4a23ff2fef7e81cf6c6", "/root/work/ladder/mse_7files.zip"),
    "group_norm": ("6ad9c2a83ff2fef7e81bf865", "/root/work/ladder/gn_7files.zip"),
    "groupnormsiluquant": ("6ad9c36f3ff2fef7e81c55cc", "/root/work/ladder/gnsq_7files.zip"),
    "index": ("6ad9c4023ff2fef7e81ca27e", "/root/work/ladder/index_7files.zip"),
    "trace": ("6aceefd6e769892927123166", "/root/work/ladder/trace_7files.zip"),
}

def req(path, data=None, method=None, timeout=180):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(BASE + path, headers=H, data=body, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try: return e.code, e.read().decode("utf-8", "replace")
        except Exception: return e.code, ""
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

def detail(sid):
    st, txt = req(f"/api/submissions/{sid}?userId={USER}")
    if st != 200:
        print(f"  detail ERR {st} {txt[:150]}"); return None
    d = json.loads(txt)
    res = d.get("result") or []
    print(f"  status={d.get('status')}  cases={len(res)}")
    for i, x in enumerate(res):
        print(f"    case{i:<2d} {str(x.get('testcase_status')):14s} prec={x.get('precision_ratio')} time={x.get('time')}")
    return d

def submit_one(name):
    pid, zp = MAP[name]
    if not os.path.exists(zp):
        print(f"{name}: ZIP MISSING {zp}"); return None
    with zipfile.ZipFile(zp) as z:
        files = [{"path": n, "content": z.read(n).decode("utf-8", "replace")}
                 for n in z.namelist() if not n.endswith("/")]
    for attempt in range(10):
        st, txt = req("/api/submissions/submit", {"problemId": pid, "userId": USER, "files": files}, method="POST")
        if st == 200:
            sid = (json.loads(txt).get("data") or {}).get("submissionId")
            print(f"{name}: submitted sid={sid} files={len(files)}")
            with open(SIDS, "a", encoding="utf-8") as f:
                f.write(f"{name}\t{pid}\t{sid}\n")
            return sid
        if st == 429:
            try: wait = json.loads(txt)["data"]["retry_after_seconds"]
            except Exception: wait = 90
            print(f"{name}: 429 wait {wait}s (try {attempt+1})", flush=True)
            time.sleep(wait + 5); continue
        print(f"{name}: SUBMIT ERR {st} {txt[:200]}"); time.sleep(20)
    return None

def poll(sid, tries=60, gap=20):
    for _ in range(tries):
        st, txt = req(f"/api/submissions/{sid}?userId={USER}", timeout=60)
        if st == 200:
            d = json.loads(txt)
            if d.get("status") not in (None, "Pending", "Running", "Queued"):
                return d
        time.sleep(gap)
    return None

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--detail":
        for sid in args[1:]:
            print(f"=== {sid}"); detail(sid)
        sys.exit(0)
    if args and args[0] == "--poll":
        for sid in args[1:]:
            print(f"=== polling {sid}", flush=True)
            d = poll(sid)
            if d: detail(sid)
            else: print("  timeout")
        sys.exit(0)
    for name in args:
        sid = submit_one(name)
        if not sid: continue
        print(f"{name}: polling...", flush=True)
        d = poll(sid)
        if d: detail(sid)
        else: print(f"{name}: poll timeout, sid={sid}")
