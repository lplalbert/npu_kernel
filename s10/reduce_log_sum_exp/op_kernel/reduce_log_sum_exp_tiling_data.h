/*!
 * \file reduce_log_sum_exp_tiling_data.h
 * \brief tiling data struct
 *
 * 注意：axes 的取值在 opapi 流程中是设备侧内存，tiling/infershape 不可解引用，
 * 因此 axes→归约映射全部由 kernel 在设备侧解析；这里只传形状类信息。
 */

#ifndef _REDUCELOGSUMEXP_TILING_DATA_H_
#define _REDUCELOGSUMEXP_TILING_DATA_H_

struct ReduceLogSumExpTilingData {
    int64_t totalNum = 0;   // 输入元素总数
    int64_t ubLen = 1024;   // 单次 UB 分块长度
    int64_t rank = 0;
    int64_t axesNum = 0;    // axes 元素个数（shape 安全可读；数据在设备侧由 kernel 自读）
    int64_t xDims[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStrides[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // 行主序步长
};
#endif
