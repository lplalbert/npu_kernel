#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CANN 天梯赛批量提交器（只提交，不做判题构建）。
用法:
  python3 ladder_submit.py --list               # 只列出待提交清单，不提交
  python3 ladder_submit.py --only trace,cmg     # 只提交指定题
  python3 ladder_submit.py                      # 按顺序提交全部
  python3 ladder_submit.py --dry-run            # 打印 payload 结构不发送
注意: 每条提交后等待冷却(~90s)并轮询结果; 配额 50/天, 次日 00:00 北京重置。
"""
import sys, os, json, time, zipfile, urllib.request, urllib.error, argparse

JWT = "<CANNJUDGE_JWT_REDACTED>.<CANNJUDGE_JWT_REDACTED>"
USER = "<MY_USER_ID>"
BASE = "https://cannjudge.cn"
CONTEST = "6aced8d2e769892927fdb17f"
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}

# name -> (problemId, zip路径)
TASKS = [
    ("flashattention",         "6acef0f7e76989292713697f", "/root/work/ladder/fa_7files.zip"),
    ("convtranspose2d",        "6acef213e769892927149a50", "/root/work/ladder/ct2d_7files.zip"),
    ("gf2poly",                "6aec3c8d1dac5d235b9a3c50", "/root/work/ladder/gf2poly_7files.zip"),
    ("gf2k_mul_square",        "6aec3a341dac5d235b9509a4", "/root/work/ladder/gf2_7files.zip"),
    ("mhcsinkhorngrad",        "6acef064e76989292712c8e3", "/root/work/ladder/msg_7files.zip"),
    ("quantmatmulsigmoidmulquant","6acef084e76989292712e8ef", "/root/work/ladder/qmsq_7files.zip"),
    ("multiaddrmsnormdynamicquant","6acef047e76989292712aaae", "/root/work/ladder/marn_official_7files.zip"),
    ("confusionmulgrad",       "6acef021e769892927127ff1", "/root/work/ladder/cmg_7files.zip"),
    ("bit_plane_transform",    "6aec3d2e1dac5d235b9ba60d", "/root/work/ladder/bpt_7files.zip"),
    ("mseloss",                "6ad9c4a23ff2fef7e81cf6c6", "/root/work/ladder/mse_7files.zip"),
    ("group_norm",             "6ad9c2a83ff2fef7e81bf865", "/root/work/ladder/gn_7files.zip"),
    ("groupnormsiluquant",     "6ad9c36f3ff2fef7e81c55cc", "/root/work/ladder/gnsq_7files.zip"),
    ("index",                  "6ad9c4023ff2fef7e81ca27e", "/root/work/ladder/index_7files.zip"),
    ("trace",                  "6aceefd6e769892927123166", "/root/work/ladder/trace_7files.zip"),
]

def req(path, data=None, method=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(BASE + path, headers=H, data=body, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try: b = e.read().decode("utf-8", "replace")
        except Exception: b = ""
        return e.code, b

def load_files(zp):
    out = []
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            out.append({"path": n, "content": z.read(n).decode("utf-8", "replace")})
    return out

def quota():
    st, b = req(f"/api/contests/{CONTEST}/my-submission-quota")
    try:
        d = json.loads(b)
        return d.get("used"), d.get("limit"), d.get("reset_at")
    except Exception:
        return None, None, b[:100]

def submit(pid, files):
    return req("/api/submissions/submit", {"problemId": pid, "userId": USER, "files": files})

def poll(sid, tries=40, gap=20):
    for _ in range(tries):
        st, b = req(f"/api/submissions/{sid}?userId={USER}")
        try:
            d = json.loads(b)
        except Exception:
            return None
        s = d.get("status")
        if s and s not in ("Pending", "Running", "Queued"):
            return d
        time.sleep(gap)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-poll", action="store_true", help="只提交不轮询（判题队列慢时用）")
    ap.add_argument("--gap", type=int, default=90, help="提交间隔秒")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(x.strip() for x in a.only.split(",") if x.strip())

    u, l, r = quota()
    print(f"quota: used={u} limit={l} reset_at={r}")
    todo = [t for t in TASKS if not only or t[0] in only]
    for name, pid, zp in todo:
        ok = os.path.exists(zp)
        nf = len(load_files(zp)) if ok else 0
        print(f"  {'OK ' if ok and nf==7 else 'BAD'} {name:30s} {pid} files={nf}")
    if a.list:
        return
    for name, pid, zp in todo:
        files = load_files(zp)
        assert len(files) == 7, f"{name}: expect 7 files, got {len(files)}"
        if a.dry_run:
            print(f"[dry] {name}: " + ", ".join(f["path"] for f in files)); continue
        st, b = submit(pid, files)
        print(f"[submit] {name} -> {st} {b[:220]}")
        if st != 200:
            if st == 429:
                print("  !! rate limited, sleep 90s"); time.sleep(90); continue
            continue
        try:
            sid = json.loads(b).get("_id") or json.loads(b).get("submissionId")
        except Exception:
            sid = None
        print(f"  submissionId={sid}")
        if sid and not a.no_poll:
            d = poll(sid)
            if d:
                res = d.get("result") or []
                okn = sum(1 for x in res if x.get("testcase_status") == "Pass")
                print(f"  -> status={d.get('status')} pass_cases={okn}/{len(res)}")
        if sid:
            open("/tmp/ladder_sids.txt", "a", encoding="utf-8").write(f"{name}\t{pid}\t{sid}\n")
        time.sleep(a.gap)

if __name__ == "__main__":
    main()
