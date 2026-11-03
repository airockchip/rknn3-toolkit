#ifndef RKNN3_LLM_CLI_LORA_H
#define RKNN3_LLM_CLI_LORA_H

#include "rknn3_api.h"
#include <string>

/* runtime state of a LoRA adapter; `loaded` and `enabled` are tracked
 * independently so load/enable/disable/unload can be driven separately:
 *   loaded  - adapter is initialized on the context (init+query+load)
 *   enabled - adapter is bound to the session (session_enable_lora) */
struct LoraState
{
  std::string    weight_path;
  rknn3_lora     lora;
  rknn3_context  ctx = 0;
  rknn3_session* session = nullptr;
  bool           loaded = false;
  bool           enabled = false;
};

/* init + query num/info + load on the context; does NOT enable on the session */
int lora_load(LoraState* st, rknn3_context ctx, const char* weight_path);

/* bind the loaded adapter to the session (session_enable_lora, clears kvcache) */
int lora_enable(LoraState* st, rknn3_session* session);

/* unbind from the session only (session_disable_lora); stays loaded */
int lora_disable(LoraState* st, rknn3_session* session);

/* unload from the context (rknn3_lora_unload); also disables if enabled */
int lora_unload(LoraState* st);

/* convenience: load + enable in one step */
int lora_setup(LoraState* st, rknn3_context ctx, rknn3_session* session, const char* weight_path);

/* re-enable an already-loaded adapter on a (re-created) session */
int lora_reapply_session(LoraState* st, rknn3_session* session);

/* detach from a session before it is destroyed (disable, keep loaded) */
int lora_detach_session(LoraState* st, rknn3_session* session);

/* convenience: disable (if enabled) + unload */
int lora_teardown(LoraState* st);

#endif /* RKNN3_LLM_CLI_LORA_H */
