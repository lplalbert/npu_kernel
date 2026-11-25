#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从指定 zip 提交某题（含 429 退避），并轮询逐 case 结果。
用法: python3 sub_zip.py <zip_path> <problemId> <tag>
"""
import os, sys, json, time, zipfile, urllib.request, urllib.error

JWT = os.environ.get("CANNJUDGE_JWT", "")
USER = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
BASE = "https://cannjudge.cn"
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}

def req(path, data=None, method=None, timeout=200):
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

zp, pid, tag = sys.argv[1:4]
with zipfile.ZipFile(zp) as z:
    files = [{"path": n, "content": z.read(n).decode("utf-8", "replace")}
             for n in z.namelist() if not n.endswith("/")]
print(f"[{tag}] zip={os.path.basename(zp)} files={len(files)}", flush=True)

sid = None
for attempt in range(10):
    st, txt = req("/api/submissions/submit", {"problemId": pid, "userId": USER, "files": files}, method="POST")
    if st == 200:
        sid = (json.loads(txt).get("data") or {}).get("submissionId")
        print(f"[{tag}] submitted sid={sid}", flush=True); break
    if st == 429:
        try: wait = json.loads(txt)["data"]["retry_after_seconds"]
        except Exception: wait = 120
        print(f"[{tag}] 429 wait {wait}s", flush=True); time.sleep(wait + 5); continue
    print(f"[{tag}] ERR {st} {txt[:200]}", flush=True); time.sleep(30)

if not sid:
    print(f"[{tag}] FAILED to submit"); sys.exit(1)

with open("/tmp/ladder_sids.txt", "a", encoding="utf-8") as f:
    f.write(f"{tag}\t{pid}\t{sid}\n")

for _ in range(80):
    st, txt = req(f"/api/submissions/{sid}?userId={USER}", timeout=60)
    if st == 200:
        d = json.loads(txt)
        if d.get("status") not in (None, "Pending", "Running", "Queued"):
            res = d.get("result") or []
            bad = [(i, x.get("precision_ratio"), x.get("time")) for i, x in enumerate(res)
                   if x.get("testcase_status") != "Pass"]
            print(f"[{tag}] {d.get('status')} total={len(res)} bad={len(bad)}", flush=True)
            for b in bad[:12]:
                print(f"   case{b[0]} prec={b[1]} time={b[2]}", flush=True)
            sys.exit(0)
    time.sleep(20)
print(f"[{tag}] poll timeout sid={sid}")
