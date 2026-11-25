#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三条线的算子交付物整理成可交接仓库（纯本地操作，不联网）。

输出：_handoff_repo/
  ladder/<op>/op_host|op_kernel/*.cpp|*.h
  ladder/zips/<op>_7files.zip
  mfg/...  s10/...
  tools/  docs/
"""
import json
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "_handoff_repo")
REVIEW = os.path.join(ROOT, "_sec5_build", "ascend_tasks", "ladder_review")

LADDER = [
    ("trace", "trace", r"_sec5_build\ascend_tasks\ladder_work\trace", "project", {}),
    ("multi_add_rms_norm_dynamic_quant", "multi_add_rms_norm_dynamic_quant", r"_sec5_build\fixa\src\marn_official", "project", {}),
    ("confusion_mul_grad", "confusion_mul_grad", r"_it6\cmg", "project", {}),
    ("bit_plane_transform", "bit_plane_transform", r"bpt_work", "flat", {"recover_md": "bpt.md", "only_missing": True}),
    ("gf2k_mul_square", "gf2k_mul_square", r"_fixb\gf2", "project", {}),
    ("gf2_poly", "gf2_poly", r"_fixb\gf2poly", "project", {}),
    ("mhc_sinkhorn_grad", "mhc_sinkhorn_grad", r"_fixc\msg", "project", {}),
    ("flash_attention", "flash_attention", None, "recover", {"recover_md": "fa.md"}),
    ("conv_transpose2_d", "conv_transpose2_d", r"ct2d_deliverable", "project", {}),
    ("quant_matmul_sigmoid_mul_quant", "quant_matmul_sigmoid_mul_quant", r"local\qmsq", "project", {}),
    ("mse_loss", "mse_loss", r"_mse_src", "project", {}),
    ("group_norm", "group_norm", r"_ladder\src\gn", "mixed", {}),
    ("group_norm_silu_quant", "group_norm_silu_quant", r"_ladder_gnsq\gnsq", "project", {}),
    ("index", "index", r"_it1_work", "zip", {}),
    ("bn_training_update", "bn_training_update", r"bn_src", "project", {}),
    ("depthwise_conv2_d", "depthwise_conv2_d", r"dwc_work\src", "project", {}),
    ("cube1", "quant_batch_mat_mul_v3", r"cube1_work\cube1", "project", {}),
    ("cube2", "quant_batch_mat_mul_v3", r"cube2_work", "project", {}),
]

MFG = [
    ("spspmm", "spspmm", r"_sec5_build\ascend_tasks\mfg_work\remote\spspmm", "project", {}),
    ("fourier_transform2d", "fourier_transform2_d", r"_sec5_build\ascend_tasks\mfg_work\fouriertransform2d", "project",
     {"extra": {"op_kernel/tiling_key_fourier_transform2_d.h":
                r"_sec5_build\ascend_tasks\mfg_work\remote\fft2d\op_kernel\tiling_key_fourier_transform2_d.h"}}),
    ("fused_prop_step", "fused_prop_step", r"mfg_fps", "flat_mfg", {}),
]

S10 = [
    ("mish", "mish", r"_sec5_build\ascend_tasks\s10_work\mish", "project", {}),
    ("reduce_log_sum_exp", "reduce_log_sum_exp", r"_sec5_build\ascend_tasks\s10_work\reduce_log_sum_exp", "project", {}),
    ("dot", "dot", r"_sec5_build\ascend_tasks\s10_work\dot", "project", {}),
    ("poisson_nll_loss", "poisson_nll_loss", r"_sec5_build\ascend_tasks\s10_work\poisson_nll_loss", "project", {}),
    ("bessel_i0e", "bessel_i0e", r"_sec5_build\ascend_tasks\s10_work\bessel_i0e", "project", {}),
]


def seven(base):
    return [("op_host", base + "_def.cpp"),
            ("op_host", base + "_infershape.cpp"),
            ("op_host", base + "_tiling.cpp"),
            ("op_kernel", base + ".cpp"),
            ("op_kernel", base + ".h"),
            ("op_kernel", base + "_tiling_data.h"),
            ("op_kernel", base + "_tiling_key.h")]


def four(base):
    return [("op_host", base + ".cpp"),
            ("op_kernel", base + ".cpp"),
            ("op_kernel", base + "_tiling.h"),
            ("op_kernel", "tiling_key_" + base + ".h")]


def parse_review(md_name):
    """从评审包 md 还原 {rel_path: content}。"""
    p = os.path.join(REVIEW, md_name)
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"## FILE (\S+)\n```cpp\n(.*?)\n```\n", txt, re.S):
        out[m.group(1)] = m.group(2)
    return out


def find_src(base_root, layout, d, f):
    if layout == "project":
        return os.path.join(base_root, d, f)
    if layout == "mixed":
        p1 = os.path.join(base_root, d, f)
        if os.path.exists(p1):
            return p1
        return os.path.join(base_root, f)
    if layout == "flat_mfg":
        if d == "op_host":
            alt = os.path.join(base_root, f[:-4] + "_host.cpp")
            if os.path.exists(alt):
                return alt
        return os.path.join(base_root, f)
    return os.path.join(base_root, f)


def emit(kind, op, base, src, layout, opts, manifest):
    dst_dir = os.path.join(REPO, kind, op)
    zips_dir = os.path.join(REPO, kind, "zips")
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(zips_dir, exist_ok=True)
    entry = {"kind": kind, "op": op, "base": base, "src": src, "layout": layout,
             "files": [], "missing": [], "zip": None, "note": ""}

    # index：直接从已有 zip 还原
    if layout == "zip":
        cand = os.path.join(ROOT, src, "index_7files_v5.zip")
        if os.path.exists(cand):
            with zipfile.ZipFile(cand) as z:
                for n in z.namelist():
                    if n.endswith("/"):
                        continue
                    dst = os.path.join(dst_dir, n.replace("/", os.sep))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    open(dst, "wb").write(z.read(n))
                    entry["files"].append(n)
            zpath = os.path.join(zips_dir, op + "_7files.zip")
            shutil.copy2(cand, zpath)
            entry["zip"] = os.path.relpath(zpath, REPO)
        else:
            entry["missing"].append("index_7files_v5.zip")
        manifest.append(entry)
        return entry

    pairs = four(base) if kind == "mfg" else seven(base)
    recovered = parse_review(opts["recover_md"]) if opts.get("recover_md") else {}
    extra = opts.get("extra", {})

    staged, missing = [], []
    for d, f in pairs:
        rel = f"{d}/{f}"
        dst = os.path.join(dst_dir, d, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        srcp = None
        if layout != "recover" and src:
            cand = find_src(os.path.join(ROOT, src), layout, d, f)
            if os.path.exists(cand):
                srcp = cand
        if srcp is None and rel in extra:
            cand = os.path.join(ROOT, extra[rel])
            if os.path.exists(cand):
                srcp = cand
        if srcp is None and rel in recovered:
            open(dst, "w", encoding="utf-8", newline="\n").write(recovered[rel] + "\n")
            entry["files"].append(rel)
            entry["note"] = entry["note"] or "部分文件从评审包 md 还原"
            staged.append((rel, dst))
            continue
        if srcp is None:
            missing.append(rel)
            continue
        shutil.copy2(srcp, dst)
        entry["files"].append(rel)
        staged.append((rel, dst))

    entry["missing"] = missing
    if not missing and staged:
        zpath = os.path.join(zips_dir, op + "_7files.zip")
        if os.path.exists(zpath):
            os.remove(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for rel, dst in staged:
                z.write(dst, arcname=rel)
        entry["zip"] = os.path.relpath(zpath, REPO)
    manifest.append(entry)
    return entry


def main():
    if os.path.isdir(REPO):
        shutil.rmtree(REPO)
    for sub in ("ladder", "mfg", "s10", "tools", "docs"):
        os.makedirs(os.path.join(REPO, sub), exist_ok=True)

    manifest = []
    for t in LADDER:
        emit("ladder", *t, manifest)
    for t in MFG:
        emit("mfg", *t, manifest)
    for t in S10:
        emit("s10", *t, manifest)

    with open(os.path.join(REPO, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"{'kind':7s} {'op':34s} {'n':3s} zip  missing")
    for e in manifest:
        print(f"{e['kind']:7s} {e['op']:34s} {len(e['files']):<3d} {'Y' if e['zip'] else 'N'}  {','.join(e['missing'])}")
    bad = [e for e in manifest if e["missing"]]
    print(f"\ntotal {len(manifest)} ops, incomplete {len(bad)}")


if __name__ == "__main__":
    main()
