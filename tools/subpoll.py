#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用：提交指定题 → 遵守 429 cooldown → 轮询到出结果 → 打印逐 case 详情。
用法:  python3 subpoll.py <name> [<name> ...]
       python3 subpoll.py --detail <submissionId>     # 只查已有提交的详情
"""
import os, sys, json, time, zipfile, urllib.request, urllib.error

JWT = os.environ.get("CANNJUDGE_JWT", "")
USER = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
BASE = "https://cannjudge.cn"
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
SIDS = "/tmp/ladder_sids.txt"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probids import LADDER, MFG, S10, zip_path


def resolve(name):
    """名字去下划线匹配（旧短名 flashattention/confusionmulgrad 等也能命中）。"""
    norm = name.replace("_", "")
    hits = []
    for kind, table in (("ladder", LADDER), ("mfg", MFG), ("s10", S10)):
        for nm, pid, *_ in table:
            if nm == name or nm.replace("_", "") == norm:
                hits.append((kind, nm, pid))
    if hits:
        return hits
    for kind, table in (("ladder", LADDER), ("mfg", MFG), ("s10", S10)):
        for nm, pid, *_ in table:
            if nm.startswith(name):
                hits.append((kind, nm, pid))
    return hits

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
    hits = resolve(name)
    if not hits:
        print(f"{name}: unknown problem"); return None
    if len(hits) > 1:
        print(f"{name}: ambiguous -> " + ", ".join(f"{k}/{n}" for k, n, _ in hits)); return None
    kind, nm, pid = hits[0]
    zp = zip_path(kind, nm)
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
