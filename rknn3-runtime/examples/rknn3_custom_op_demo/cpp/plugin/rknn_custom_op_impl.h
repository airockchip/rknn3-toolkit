/*
 * rknn_custom_op_impl.h - Custom operator function declarations.
 *
 * Include this header in rknn3_custom_op.c instead of forward declarations
 * to ensure proper linking when loaded via dlopen_v2 on RT-Thread.
 */

#ifndef _RKNN_CUSTOM_OP_IMPL_H
#define _RKNN_CUSTOM_OP_IMPL_H

#include "rknn3_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* RknnCustomOpExam - custom op compute callback */
int rknn_custom_op_compute(rknn3_custom_op_context *op_ctx,
                           rknn3_tensor *inputs, uint32_t n_inputs,
                           rknn3_tensor *outputs, uint32_t n_outputs);

/* RknnPostProcess - postprocess op callbacks */
int rknn_postprocess_compute(rknn3_custom_op_context *op_ctx,
                             rknn3_tensor *inputs, uint32_t n_inputs,
                             rknn3_tensor *outputs, uint32_t n_outputs);

int rknn_postprocess_get_attrs(rknn3_custom_op_context *op_ctx,
                               rknn3_tensor_attr *input_attrs, uint32_t n_inputs,
                               rknn3_tensor_attr *output_attrs, uint32_t n_outputs);

int rknn_postprocess_get_output_num(rknn3_custom_op_context *op_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _RKNN_CUSTOM_OP_IMPL_H */
