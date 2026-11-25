#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三大判题赛的题目名 -> problemId 映射（名字与仓库目录名一致）。

天梯赛 / mfg / S10 共用 cannjudge.cn 判题平台，只有题目与文件结构不同：
  ladder: 7 文件包，contest 6aced8d2e769892927fdb17f，50 次/天，截止 11-30 17:00
  mfg:    3/4 文件包，contest 6aed4926a534d8cef5dd8d0c，50 次/天，截止 12-07 12:00
  s10:    7 文件包，10 次/天
"""
import os

CONTEST_LADDER = "6aced8d2e769892927fdb17f"
CONTEST_MFG = "6aed4926a534d8cef5dd8d0c"

LADDER = [
    # (仓库目录名, problemId, 难度)
    ("trace", "6aceefd6e769892927123166", 1),
    ("multi_add_rms_norm_dynamic_quant", "6acef047e76989292712aaae", 2),
    ("confusion_mul_grad", "6acef021e769892927127ff1", 2),
    ("bit_plane_transform", "6aec3d2e1dac5d235b9ba60d", 2),
    ("gf2k_mul_square", "6aec3a341dac5d235b9509a4", 3),
    ("gf2_poly", "6aec3c8d1dac5d235b9a3c50", 4),
    ("mhc_sinkhorn_grad", "6acef064e76989292712c8e3", 3),
    ("flash_attention", "6acef0f7e76989292713697f", 5),
    ("conv_transpose2_d", "6acef213e769892927149a50", 5),
    ("quant_matmul_sigmoid_mul_quant", "6acef084e76989292712e8ef", 3),
    ("mse_loss", "6ad9c4a23ff2fef7e81cf6c6", 2),
    ("group_norm", "6ad9c2a83ff2fef7e81bf865", 2),
    ("group_norm_silu_quant", "6ad9c36f3ff2fef7e81c55cc", 2),
    ("index", "6ad9c4023ff2fef7e81ca27e", 2),
    ("bn_training_update", "6ad9c0583ff2fef7e81ab6bb", 1),
    ("depthwise_conv2_d", "6ad9c1ff3ff2fef7e81b9d0e", 5),
    ("cube1", "6ad9c71b3ff2fef7e81e2f75", 4),
    ("cube2", "6ad9c7f83ff2fef7e81e894c", 4),
]

MFG = [
    # (仓库目录名, problemId)
    ("ball_query", "6af1c92b6bd0f9be3798a712"),
    ("complex_abs", "6af2d3706bd0f9be375466ed"),
    ("complex_div", "6af2d43a6bd0f9be3756251e"),
    ("complex_exp", "6af6cbd36bd0f9be37ba0aae"),
    ("cylinder_query", "6af1c67a6bd0f9be3792fb4a"),
    ("fourier_transform2d", "6af2d6646bd0f9be375ab041"),
    ("furthest_point_sample", "6af1c8736bd0f9be37971169"),
    ("fused_prop_step", "6af2d5b76bd0f9be3759446c"),
    ("gather_coo", "6af1cbe56bd0f9be379e1c23"),
    ("grid_cluster", "6af1cab46bd0f9be379bc6c6"),
    ("radius", "6af1ca066bd0f9be379a6c84"),
    ("random_walk", "6af277846bd0f9be37a6755a"),
    ("segment_min_csr", "6af1cdaa6bd0f9be37a1afe1"),
    ("spmm_sum", "6af275c16bd0f9be37a3bbd5"),
    ("spspmm", "6af276736bd0f9be37a4cb08"),
]

S10 = [
    ("mish", "6ad9cfab3ff2fef7e8227643"),
    ("bessel_i0e", "6ad9d3523ff2fef7e82428fe"),
    ("dot", "6ad9d4bb3ff2fef7e824d520"),
    ("poisson_nll_loss", "6ad9d60b3ff2fef7e8257053"),
    ("reduce_log_sum_exp", "6ad9d7423ff2fef7e82610ca"),
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def zip_path(kind, name):
    return os.path.join(REPO_ROOT, kind, "zips", f"{name}_7files.zip")
