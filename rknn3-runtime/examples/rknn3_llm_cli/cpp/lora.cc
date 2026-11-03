// LoRA adapter support for rknn3_llm_cli.
// The RKNN3 LoRA flow is context-scoped (init/query/load/unload) plus
// session-scoped (enable/disable):
//   lora_init -> query LORA_NUM -> query LORA_INFO -> lora_load
//   -> session_enable_lora  (per session; re-apply when a session is recreated)
// Load and enable (and disable and unload) are exposed separately so callers
// can drive the lifecycle step by step.

#include "lora.h"

#include <cstdio>
#include <cstring>

int lora_load(LoraState* st, rknn3_context ctx, const char* weight_path)
{
  if (!st || !ctx || !weight_path || weight_path[0] == '\0')
    return -1;

  /* the same adapter is already loaded: nothing to do */
  if (st->loaded && st->weight_path == weight_path)
    return 0;

  /* switching adapters: release the previous one first. A failed unload
     leaves the old adapter on the context, so loading the new one must be
     aborted to keep CLI state consistent with the runtime. */
  if (st->loaded)
  {
    int uret = lora_unload(st);
    if (uret < 0)
    {
      printf("warning: previous LoRA unload failed, ret=%d; aborting switch "
             "(context may hold stale lora)\n", uret);
      return uret;
    }
  }

  rknn3_lora lora_list[RKNN3_MAX_LORA_NUM];
  memset(lora_list, 0, sizeof(lora_list));

  int ret = rknn3_lora_init(ctx, weight_path);
  if (ret < 0)
  {
    printf("rknn3_lora_init failed, ret=%d\n", ret);
    return ret;
  }

  uint32_t n_lora = 0;
  ret = rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora));
  if (ret < 0)
  {
    printf("rknn3_query LORA_NUM failed, ret=%d (lora_init state remains on context)\n", ret);
    return ret;
  }
  if (n_lora == 0)
  {
    printf("no lora adapter found in '%s' (lora_init state remains on context)\n", weight_path);
    return -1;
  }

  ret = rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list));
  if (ret < 0)
  {
    printf("rknn3_query LORA_INFO failed, ret=%d (lora_init state remains on context)\n", ret);
    return ret;
  }

  ret = rknn3_lora_load(ctx, &lora_list[0]);
  if (ret < 0)
  {
    printf("rknn3_lora_load failed, ret=%d\n", ret);
    return ret;
  }

  st->weight_path = weight_path;
  st->lora = lora_list[0];
  st->ctx = ctx;
  st->loaded = true;
  st->enabled = false;
  return 0;
}

int lora_enable(LoraState* st, rknn3_session* session)
{
  if (!st || !session)
    return -1;
  if (!st->loaded)
  {
    printf("no LoRA loaded, use /lora <path> first\n");
    return -1;
  }
  int ret = rknn3_session_enable_lora(session, &st->lora);
  if (ret < 0)
  {
    printf("rknn3_session_enable_lora failed, ret=%d\n", ret);
    return ret;
  }
  st->session = session;
  st->enabled = true;
  return 0;
}

/* returns 0 on success, 1 when the adapter was not enabled (no-op), <0 on error */
int lora_disable(LoraState* st, rknn3_session* session)
{
  if (!st)
    return -1;
  if (!st->enabled)
    return 1;
  if (!session)
    return -1;
  int ret = rknn3_session_disable_lora(session, &st->lora);
  if (ret < 0)
  {
    printf("rknn3_session_disable_lora failed, ret=%d (adapter stays enabled)\n", ret);
    return ret;
  }
  st->enabled = false;
  return 0;
}

int lora_unload(LoraState* st)
{
  if (!st)
    return -1;

  /* disable first if still enabled; a disable failure means the adapter is
     still bound to the session, so do NOT unload it (state must stay). */
  if (st->enabled)
  {
    int dret = lora_disable(st, st->session);
    if (dret < 0)
    {
      printf("lora_unload: disable failed, ret=%d (adapter left loaded+enabled)\n", dret);
      return dret;
    }
  }

  int ret = 0;
  if (st->loaded && st->ctx)
  {
    ret = rknn3_lora_unload(st->ctx, &st->lora);
    if (ret < 0)
    {
      printf("rknn3_lora_unload failed, ret=%d (adapter left loaded)\n", ret);
      return ret;
    }
  }
  st->loaded = false;
  st->enabled = false;
  st->weight_path.clear();
  return 0;
}

int lora_setup(LoraState* st, rknn3_context ctx, rknn3_session* session, const char* weight_path)
{
  if (!st || !ctx || !session || !weight_path || weight_path[0] == '\0')
    return -1;
  if (lora_load(st, ctx, weight_path) != 0)
    return -1;
  if (lora_enable(st, session) != 0)
  {
    lora_unload(st);
    return -1;
  }
  return 0;
}

int lora_reapply_session(LoraState* st, rknn3_session* session)
{
  if (!st || !st->loaded || !session)
    return 0;
  return lora_enable(st, session);
}

int lora_detach_session(LoraState* st, rknn3_session* session)
{
  return lora_disable(st, session);
}

int lora_teardown(LoraState* st)
{
  return lora_unload(st);
}
