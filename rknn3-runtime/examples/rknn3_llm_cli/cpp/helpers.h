// Shared types / helpers / globals for the rknn3 LLM demos.
//
// This is the common core used by both front-ends:
//   - rknn3_llm_cli            (interactive CLI, internal session decode)
//   - rknn3_llm_step_decode    (per-token / external decode loop)
//
// The inference engine (session init, callbacks, run_turn) lives in engine.cc
// and drives a single shared SessionState `g_state`; the only difference
// between the two front-ends is the decode strategy selected in
// SessionState::decode_mode (see engine.h).
#ifndef RKNN3_LLM_CLI_HELPERS_H
#define RKNN3_LLM_CLI_HELPERS_H

#include "Tokenizer.h"
#include "float16.h"
#include "rknn3_api.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

/*=============================================================================
   Logging
=============================================================================*/

#define LOGW(fmt, ...) printf("\033[33m" fmt "\033[0m", ##__VA_ARGS__)
#define LOGI(fmt, ...) printf("\033[36m" fmt "\033[0m", ##__VA_ARGS__)
#define LOGR(fmt, ...) printf("\033[31m" fmt "\033[0m", ##__VA_ARGS__)
#define LOGG(fmt, ...) printf("\033[32m" fmt "\033[0m", ##__VA_ARGS__)
#define VLOG(fmt, ...) do { if (g_state.verbose) printf(fmt, ##__VA_ARGS__); } while (0)

/*=============================================================================
   Data structures
=============================================================================*/

struct embedding_info
{
  int      fd = -1;
  float16* embedding_data = nullptr;
  int      embedding_dim = 0;
  int      vocab_size = 0;
  size_t   data_size = 0;
};

/* performance statistics of a single generation turn */
struct RunStats
{
  bool     valid = false;
  int64_t  ttft_us = 0;              /* time to first token */
  double   prefill_ms = 0.0;
  double   decode_ms = 0.0;
  uint64_t n_prefill_tokens = 0;
  uint64_t n_decode_tokens = 0;
  double   prefill_tps = 0.0;
  double   decode_tps = 0.0;
  double   prefill_tpt_ms = 0.0;
  double   decode_tpt_ms = 0.0;
  int      stopped_by_user = 0;
};

/* last-token result of a single inference step, produced by the result
   callback and consumed by the external (per-token) decode loop. */
struct StageResultState
{
  std::mutex mutex;
  bool       has_token = false;
  int32_t    next_token = -1;
};

/*=============================================================================
   Small helpers (inline so all modules can use them)
=============================================================================*/

inline int64_t get_current_time_us()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

inline double elapsed_us(const timeval& start, const timeval& end)
{
  return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
}

inline double elapsed_ms(const timeval& begin, const timeval& end)
{
  return (double)(end.tv_sec - begin.tv_sec) * 1000.0 +
         (double)(end.tv_usec - begin.tv_usec) / 1000.0;
}

inline std::string trim(const std::string& s)
{
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

inline bool iequals(const std::string& a, const std::string& b)
{
  return strcasecmp(a.c_str(), b.c_str()) == 0;
}

inline std::vector<std::string> split_args(const std::string& line)
{
  std::vector<std::string> args;
  std::string cur;
  for (size_t i = 0; i < line.size(); ++i)
  {
    char c = line[i];
    if (c == ' ' || c == '\t')
    {
      if (!cur.empty()) { args.push_back(cur); cur.clear(); }
    }
    else
    {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) args.push_back(cur);
  return args;
}

/* read one line from stdin; returns chars read, -1 on EOF. Handles any length. */
inline ssize_t read_line(std::string& line)
{
  line.clear();
  int c;
  while ((c = fgetc(stdin)) != EOF)
  {
    if (c == '\n') return (ssize_t)line.size();
    line.push_back((char)c);
  }
  return line.empty() ? -1 : (ssize_t)line.size();
}

/* read an entire text file into a string */
inline bool read_file(const char* path, std::string& out)
{
  FILE* fp = fopen(path, "rb");
  if (!fp) return false;
  char buf[4096];
  size_t n;
  out.clear();
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
  fclose(fp);
  return true;
}

/* true when path names an existing regular file (not a dir / device) */
inline bool file_exists(const char* path)
{
  struct stat st;
  return path && path[0] != '\0' && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

#endif /* RKNN3_LLM_CLI_HELPERS_H */
