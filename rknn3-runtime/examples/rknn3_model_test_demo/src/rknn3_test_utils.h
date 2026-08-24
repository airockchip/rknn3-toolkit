#ifndef __RKNN3_TEST_UTILS_H__
#define __RKNN3_TEST_UTILS_H__

#include "rknn3_api.h"

#include <stdint.h>
#include <stdio.h>
#include <vector>

int setup_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* kvcache_mems);
void cleanup_kvcache_mems(rknn3_context ctx, std::vector<rknn3_tensor_mem*>* mems);

#endif
