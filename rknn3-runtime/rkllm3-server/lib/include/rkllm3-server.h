#ifndef _RKLLM3_SERVER_H
#define _RKLLM3_SERVER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SERVER_MODEL_INITED       = 0,
  SERVER_EXITED             = 1,
  SERVER_ERROR              = 2,
} ServerStatus;

typedef void (*status_callback_t)(void* userdata, ServerStatus status);

typedef struct {
  status_callback_t status_callback;
  void*             status_userdata;
} StatusCallback;


int start_server(const char *json, StatusCallback *callback);
int stop_server();

#ifdef __cplusplus
}   // extern "C"
#endif

#endif // _RKLLM3_SERVER_H