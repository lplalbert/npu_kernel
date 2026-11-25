#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交 mfg 题目（从仓库 mfg/zips/ 读取 3/4 文件包），处理 429 冷却，并轮询结果。
用法:  python3 mfg_submit3.py [name ...]   # 不传名字 = 全部 15 题
凭据从环境变量读: CANNJUDGE_JWT / CANNJUDGE_USER（见 tools/CREDENTIALS.md）。
"""
import json, os, re, subprocess, sys, time, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probids import MFG, zip_path

JWT = os.environ.get("CANNJUDGE_JWT", "")
COOKIE = f"cannjudge_auth={JWT}"
USER_ID = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
BASE = "https://cannjudge.cn/api"

PROBS = [(name, pid, zip_path("mfg", name)) for name, pid in MFG]


def curl(args):
    p = subprocess.run(["curl", "-s", "-S", "-w", "\n__H__%{http_code}"] + args,
                       capture_output=True, text=True, encoding="utf-8")
    out = p.stdout or ""
    m = re.search(r"\n__H__(\d+)\s*$", out)
    code = int(m.group(1)) if m else None
    if m:
        out = out[:m.start()]
    return code, out.strip()


def submit(name, pid, zp):
    if not os.path.exists(zp):
        print(f"{name}: ZIP MISSING {zp}"); return None
    with zipfile.ZipFile(zp) as z:
        files = [{"path": n, "content": z.read(n).decode("utf-8", "replace")}
                 for n in z.namelist() if not n.endswith("/")]
    payload = {"problemId": pid, "userId": USER_ID, "files": files}
    body = json.dumps(payload, ensure_ascii=False)
    tmp = f"/tmp/mfg_{name}.json"
    open(tmp, "w", encoding="utf-8").write(body)
    print(f"{name}: {len(files)} files, {len(body)} bytes", flush=True)
    for attempt in range(8):
        code, out = curl(["-X", "POST", f"{BASE}/submissions/submit",
                          "-H", f"Cookie: {COOKIE}", "-H", "Content-Type: application/json",
                          "--data-binary", f"@{tmp}"])
        if code == 200:
            try:
                sid = (json.loads(out).get("data") or {}).get("submissionId")
            except Exception:
                sid = None
            print(f"{name}: OK sid={sid}", flush=True)
            with open("/tmp/mfg_sids.txt", "a", encoding="utf-8") as f:
                f.write(f"{name}\t{pid}\t{sid}\n")
            return sid
        if code == 429:
            try: wait = json.loads(out)["data"]["retry_after_seconds"]
            except Exception: wait = 120
            print(f"{name}: 429 wait {wait}s", flush=True); time.sleep(wait + 5); continue
        print(f"{name}: ERR {code} {out[:200]}", flush=True); time.sleep(20)
    return None


def poll(sid, tries=80, gap=20):
    for _ in range(tries):
        code, out = curl([f"{BASE}/submissions/{sid}?userId={USER_ID}",
                          "-H", f"Cookie: {COOKIE}"])
        if code == 200:
            d = json.loads(out)
            if d.get("status") not in (None, "Pending", "Running", "Queued"):
                return d
        time.sleep(gap)
    return None


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    norm = {w.replace("_", "") for w in want}
    for name, pid, zp in PROBS:
        if want and name not in want and name.replace("_", "") not in norm:
            continue
        sid = submit(name, pid, zp)
        if not sid: continue
        print(f"{name}: polling...", flush=True)
        d = poll(sid)
        if not d:
            print(f"{name}: poll timeout"); continue
        res = d.get("result") or []
        bad = [(i, x.get("testcase_status"), x.get("precision_ratio"), x.get("time"))
               for i, x in enumerate(res) if x.get("testcase_status") != "Pass"]
        print(f"{name}: {d.get('status')} total={len(res)} bad={len(bad)}", flush=True)
        for b in bad[:10]:
            print(f"   case{b[0]} {b[1]} prec={b[2]} time={b[3]}", flush=True)
        time.sleep(15)
    print("MFG DONE", flush=True)
