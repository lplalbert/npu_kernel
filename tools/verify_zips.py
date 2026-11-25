#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验仓库里每个 zip 的成员是否与对应源码目录逐字节一致（提交前必跑）。"""
import os, sys, zipfile, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probids import LADDER, MFG, S10, zip_path

EXPECT = {"ladder": 7, "s10": 7}


def md5(b):
    return hashlib.md5(b).hexdigest()


bad_total = 0
for kind, table, exp in (("ladder", LADDER, 7), ("mfg", MFG, None), ("s10", S10, 7)):
    for name, pid, *_ in table:
        zp = zip_path(kind, name)
        src = os.path.join(os.path.dirname(os.path.dirname(zp)), name)
        if not os.path.exists(zp):
            print(f"{kind:7s} {name:26s} ZIP-MISSING"); bad_total += 1; continue
        try:
            with zipfile.ZipFile(zp) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
                diffs = []
                for n in names:
                    sp = os.path.join(src, n.replace("/", os.sep))
                    if not os.path.exists(sp):
                        diffs.append(f"{n}:SRC-MISSING"); continue
                    if md5(z.read(n)) != md5(open(sp, "rb").read()):
                        diffs.append(n)
                nok = len(names) == exp if exp else len(names) in (3, 4)
                status = "OK" if (nok and not diffs) else "MISMATCH"
                if status != "OK":
                    bad_total += 1
                print(f"{kind:7s} {name:26s} {status:9s} entries={len(names)} diffs={diffs if diffs else '-'}")
        except Exception as e:
            print(f"{kind:7s} {name:26s} ERR {e}"); bad_total += 1

print("\n=== 不一致/缺失总数:", bad_total, "===")
