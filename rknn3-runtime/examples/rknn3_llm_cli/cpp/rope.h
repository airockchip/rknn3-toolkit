// External rope cache support for the rknn3 LLM demos.
// Some models (Qwen3.5 / Gemma-4 with rope_cache_host_storage=1) keep the
// rope cos/sin caches on the host and feed them per generation step through
// the session's input_callback.
#ifndef RKNN3_LLM_CLI_ROPE_H
#define RKNN3_LLM_CLI_ROPE_H

#include "rknn3_api.h"

#include <cstddef>
#include <cstdint>

/* one external rope cache tensor loaded from a rope cache file */
struct rope_cache_tensor
{
  void*              data = nullptr;
  size_t             data_size = 0; /* tensor byte size = prod(shape) * elem */
  int                n_dims = 0;
  int                shape[5] = {};
  int                dtype = 0;
  int                layout = 0;
};

enum class RopeCacheFormat
{
  NONE = 0,
  GEMMA4, /* rope_cos_cache_0/1, rope_sin_cache_0/1 */
  QWEN35, /* rope_cos_cache,     rope_sin_cache      */
};

/* runtime state for external rope caches (mmap'd rope cache + callback) */
struct RopeState
{
  rope_cache_tensor rope_caches[4];
  RopeCacheFormat   rope_format = RopeCacheFormat::NONE;
  int*              ext_input_indices = nullptr; /* indices of rope / per_layer inputs */
  int               n_ext_inputs = 0;
  int               rope_fd = -1;
  void*             rope_mmap_base = nullptr;
  size_t            rope_mmap_size = 0;
  bool              enabled = false;
};

/* load the rope caches from a rope cache file (Qwen3.5 / Gemma-4 formats),
 * mmap the file, and record the rope input-tensor indices from the model. */
int rope_setup(RopeState* st, rknn3_context ctx, const char* rope_path);

/* input_callback used by the runtime to fill rope caches per generation step */
int rope_input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                        LLMInputCallbackParam param);

/* munmap the rope cache file and free the ext-input index array */
void rope_teardown(RopeState* st);

#endif /* RKNN3_LLM_CLI_ROPE_H */
