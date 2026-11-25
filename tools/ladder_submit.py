#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CANN 天梯赛批量提交器（只提交，不做判题构建）。
用法:
  python3 ladder_submit.py --list               # 只列出待提交清单，不提交
  python3 ladder_submit.py --only trace,cmg     # 只提交指定题（名字去下划线匹配）
  python3 ladder_submit.py                      # 按顺序提交全部
  python3 ladder_submit.py --dry-run            # 打印 payload 结构不发送
注意: 每条提交后等待冷却(~90s)并轮询结果; 配额 50/天, 次日 00:00 北京重置。
凭据从环境变量读: CANNJUDGE_JWT / CANNJUDGE_USER（见 tools/CREDENTIALS.md）。
"""
import sys, os, json, time, zipfile, urllib.request, urllib.error, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probids import LADDER, CONTEST_LADDER, zip_path

JWT = os.environ.get("CANNJUDGE_JWT", "")
USER = os.environ.get("CANNJUDGE_USER", "<MY_USER_ID>")
BASE = "https://cannjudge.cn"
CONTEST = CONTEST_LADDER
H = {"Cookie": f"cannjudge_auth={JWT}", "Content-Type": "application/json",
     "Accept": "application/json", "User-Agent": "Mozilla/5.0"}

# name -> (problemId, zip路径)
TASKS = [(name, pid, zip_path("ladder", name)) for name, pid, _ in LADDER]


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
    only = {x.strip().replace("_", "") for x in a.only.split(",") if x.strip()}

    u, l, r = quota()
    print(f"quota: used={u} limit={l} reset_at={r}")
    todo = [t for t in TASKS if not only or t[0].replace("_", "") in only]
    for name, pid, zp in todo:
        ok = os.path.exists(zp)
        nf = len(load_files(zp)) if ok else 0
        print(f"  {'OK ' if ok and nf == 7 else 'BAD'} {name:34s} {pid} files={nf}")
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
