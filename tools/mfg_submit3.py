#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交 mfg 三道题（4 文件结构），处理 429 冷却，并轮询结果。"""
import json, os, re, subprocess, sys, time

COOKIE = ("cannjudge_auth=<CANNJUDGE_JWT_REDACTED>"
          ".<CANNJUDGE_JWT_REDACTED>")
USER_ID = "<MY_USER_ID>"
BASE = "https://cannjudge.cn/api"

PROBS = [
    ("spspmm", "6af276736bd0f9be37a4cb08", "/root/work/mfg/spspmm",
     ["op_host/spspmm.cpp", "op_kernel/spspmm.cpp",
      "op_kernel/spspmm_tiling.h", "op_kernel/tiling_key_spspmm.h"]),
    ("fourier_transform2d", "6af2d6646bd0f9be375ab041", "/root/work/mfg/fouriertransform2d",
     ["op_host/fourier_transform2_d.cpp", "op_kernel/fourier_transform2_d.cpp",
      "op_kernel/fourier_transform2_d_tiling.h", "op_kernel/tiling_key_fourier_transform2_d.h"]),
    ("fused_prop_step", "6af2d5b76bd0f9be3759446c", "/root/work/mfg/fusedpropstep",
     ["op_host/fused_prop_step.cpp", "op_kernel/fused_prop_step.cpp",
      "op_kernel/fused_prop_step_tiling.h", "op_kernel/tiling_key_fused_prop_step.h"]),
]

def curl(args):
    p = subprocess.run(["curl", "-s", "-S", "-w", "\n__H__%{http_code}"] + args,
                       capture_output=True, text=True, encoding="utf-8")
    out = p.stdout or ""
    m = re.search(r"\n__H__(\d+)\s*$", out)
    code = int(m.group(1)) if m else None
    if m:
        out = out[:m.start()]
    return code, out.strip()

def submit(name, pid, root, rels):
    files = []
    for rel in rels:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"{name}: MISSING {p}"); return None
        files.append({"path": rel, "content": open(p, encoding="utf-8", errors="replace").read()})
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
    for name, pid, root, rels in PROBS:
        if want and name not in want:
            continue
        sid = submit(name, pid, root, rels)
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
