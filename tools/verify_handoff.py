#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 _handoff_repo 下全部 zip：条目数、路径前缀、与源码 md5 一致、禁词、非 ASCII。"""
import hashlib
import json
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_handoff_repo")
# 真正会触发判题「不合规内容」的：std::cout/cin 系 与 函数调用形式的调试/系统接口
BAD = re.compile(r"std::(cout|cin|cerr|clog)|(?<![\w.])(printf|fprintf|sprintf|snprintf|getenv|system|popen)\s*\(")
# 仅提示：把 cout/cin 当变量名（历史上有 400 案例，建议改名 chOut/chIn）
WARN = re.compile(r"\b(cout|cin)\b")
EXPECT = {"ladder": 7, "mfg": 4, "s10": 7}

rows, problems = [], []
for kind in ("ladder", "mfg", "s10"):
    zdir = os.path.join(REPO, kind, "zips")
    for zn in sorted(os.listdir(zdir)):
        if not zn.endswith(".zip"):
            continue
        op = zn[:-len("_7files.zip")]
        zp = os.path.join(zdir, zn)
        with __import__("zipfile").ZipFile(zp) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            md5ok, badcnt, nonascii, warncnt = 0, 0, 0, 0
            for n in names:
                data = z.read(n)
                srcp = os.path.join(REPO, kind, op, n.replace("/", os.sep))
                if os.path.exists(srcp) and hashlib.md5(data).hexdigest() == hashlib.md5(open(srcp, "rb").read()).hexdigest():
                    md5ok += 1
                txt = data.decode("utf-8", "replace")
                badcnt += len(BAD.findall(txt))
                warncnt += len(WARN.findall(txt))
                nonascii += sum(1 for c in txt if ord(c) > 127)
        ok = (len(names) == EXPECT[kind] and md5ok == len(names) and badcnt == 0)
        rows.append((kind, op, len(names), md5ok, badcnt, warncnt, nonascii, "OK" if ok else "FAIL"))
        if not ok:
            problems.append((kind, op, len(names), md5ok, badcnt))

print(f"{'kind':7s} {'op':34s} {'n':3s} {'md5':4s} {'bad':4s} {'warn':5s} {'nonA':5s} status")
for r in rows:
    print(f"{r[0]:7s} {r[1]:34s} {r[2]:<3d} {r[3]:<4d} {r[4]:<4d} {r[5]:<5d} {r[6]:<5d} {r[7]}")
print(f"\nzip 总数 {len(rows)}，不合格 {len(problems)}")
if problems:
    for p in problems:
        print("  FAIL:", p)
sys.exit(1 if problems else 0)
