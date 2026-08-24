#include "cnpy.h"

#include <stdio.h>
#include <string.h>
#include <cstdlib>

// numpy文件头魔数
static const char    magic_string[] = "\x93NUMPY";
static const uint8_t magic_len      = 6;

// numpy头部版本
static const uint8_t major_version = 1;
static const uint8_t minor_version = 0;

// // 检查文件是否是npy格式
// static bool is_npy_file(const char *filepath) {
//   const char *ext = strrchr(filepath, '.');
//   return ext && strcmp(ext, ".npy") == 0;
// }

/**
 * @brief 从字符串中解析shape信息
 * @param str: 包含shape信息的字符串，格式如"(100,200)"或"(1000,)"
 * @param arr: numpy数组结构体
 * @return 成功返回0，失败返回-1
 */
static int parse_shape(const char* str, npy_array* arr)
{
  const char* p = str;
  size_t      dims[32]; // 最大支持32维
  size_t      dim_count = 0;

  // 跳过开始的'('
  if (*p++ != '(') {
    return -1;
  }

  // 解析每个维度
  while (*p && *p != ')') {
    char* end;
    dims[dim_count] = strtoul(p, &end, 10);
    if (p == end) { // 转换失败
      return -1;
    }
    dim_count++;
    p = end;
    if (*p == ',')
      p++; // 跳过逗号
  }

  // 分配shape内存
  arr->shape = (size_t*)malloc(dim_count * sizeof(size_t));
  if (!arr->shape) {
    return -1;
  }

  // 复制shape信息
  arr->ndim = dim_count;
  memcpy(arr->shape, dims, dim_count * sizeof(size_t));

  return 0;
}

/**
 * @brief 从字符串中解析数据类型
 * @param str: 类型描述字符串，如"<f4"、"<i4"等
 * @return 对应的cnpy_type枚举值
 */
static cnpy_type parse_dtype(const char* str)
{
  // 跳过字节序标记
  if (*str == '<' || *str == '>' || *str == '|') {
    str++;
  }

  // 解析类型
  switch (*str) {
  case 'f':
    if (str[1] == '4')
      return CNPY_TYPE_FLOAT32;
    if (str[1] == '8')
      return CNPY_TYPE_FLOAT64;
    break;
  case 'i':
    if (str[1] == '1')
      return CNPY_TYPE_INT8;
    if (str[1] == '2')
      return CNPY_TYPE_INT16;
    if (str[1] == '4')
      return CNPY_TYPE_INT32;
    if (str[1] == '8')
      return CNPY_TYPE_INT64;
    break;
  case 'u':
    if (str[1] == '1')
      return CNPY_TYPE_UINT8;
    if (str[1] == '2')
      return CNPY_TYPE_UINT16;
    if (str[1] == '4')
      return CNPY_TYPE_UINT32;
    if (str[1] == '8')
      return CNPY_TYPE_UINT64;
    break;
  case 'b':
    return CNPY_TYPE_BOOLEAN;
    break;
  }

  printf("Invalid numpy dtype %s\n", str);

  return CNPY_TYPE_FLOAT32; // 默认返回float32
}

/**
 * @brief 解析numpy头部信息
 * @param fp: 文件指针
 * @param arr: numpy数组结构体
 * @return 成功返回0，失败返回-1
 */
static int parse_npy_header(FILE* fp, npy_array* arr)
{
  char     buffer[256];
  uint16_t header_len;
  char *   shape_start, *shape_end;
  char *   dtype_start, *dtype_end;
  char     shape_str[64] = {0};
  char     dtype_str[8]  = {0};

  // 检查魔数
  if (fread(buffer, 1, magic_len, fp) != magic_len) {
    return -1;
  }
  if (memcmp(buffer, magic_string, magic_len) != 0) {
    return -1;
  }

  // 检查版本
  if (fread(buffer, 1, 2, fp) != 2) {
    return -1;
  }
  if (buffer[0] != major_version || buffer[1] != minor_version) {
    return -1;
  }

  // 读取头部长度
  if (fread(&header_len, sizeof(header_len), 1, fp) != 1) {
    return -1;
  }

  // 读取头部信息
  if (fread(buffer, 1, header_len, fp) != header_len) {
    return -1;
  }
  buffer[header_len] = '\0';

  // 解析数据类型
  dtype_start = strstr(buffer, "'descr': '");
  if (dtype_start) {
    dtype_start += 10; // 跳过"'descr': '"
    dtype_end = strchr(dtype_start, '\'');
    if (dtype_end && (dtype_end - dtype_start) < (int)sizeof(dtype_str)) {
      strncpy(dtype_str, dtype_start, dtype_end - dtype_start);
      arr->dtype = parse_dtype(dtype_str);
    }
  }

  // 解析数组序
  arr->fortran_order = (strstr(buffer, "'fortran_order': True") != NULL);

  // 解析shape
  shape_start = strstr(buffer, "'shape': (");
  if (shape_start) {
    shape_start += 9; // 跳过"'shape': "
    shape_end = strchr(shape_start, ')');
    if (shape_end && (shape_end - shape_start + 1) < (int)sizeof(shape_str)) {
      strncpy(shape_str, shape_start, shape_end - shape_start + 1);
      if (parse_shape(shape_str, arr) != 0) {
        return -1;
      }
    }
  }

  // 记录数据开始位置
  arr->data_begin = ftell(fp);

  return 0;
}

/**
 * @brief 打开numpy文件
 */
int npy_open(const char* filepath, npy_array* arr)
{
  FILE*  fp;
  size_t file_size;

  if (!filepath || !arr) {
    return -1;
  }

  // 初始化结构体
  memset(arr, 0, sizeof(npy_array));

  // 打开文件
  fp = fopen(filepath, "rb");
  if (!fp) {
    return -1;
  }

  // 获取文件大小
  fseek(fp, 0, SEEK_END);
  file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  // 分配内存并读取整个文件
  arr->raw_data = malloc(file_size);
  if (!arr->raw_data) {
    fclose(fp);
    return -1;
  }
  arr->raw_data_size = file_size;

  if (fread(arr->raw_data, 1, file_size, fp) != file_size) {
    free(arr->raw_data);
    fclose(fp);
    return -1;
  }

  // 在这里添加：重置文件指针到文件开始处
  fseek(fp, 0, SEEK_SET);

  // 解析头部信息
  if (parse_npy_header(fp, arr) != 0) {
    free(arr->raw_data);
    fclose(fp);
    return -1;
  }

  fclose(fp);
  return 0;
}

/**
 * @brief 关闭numpy数组，释放资源
 */
void npy_close(npy_array* arr)
{
  if (arr) {
    if (arr->raw_data) {
      free(arr->raw_data);
    }
    if (arr->shape) {
      free(arr->shape);
    }
    memset(arr, 0, sizeof(npy_array));
  }
}

/**
 * @brief 保存float数据到numpy文件
 */
int npy_save_float_buffer_to_file(const char* filepath, float* data, size_t size, size_t* shape, size_t ndim)
{
  FILE*    fp;
  char     header[256];
  uint16_t header_len;

  if (!filepath || !data || !shape || ndim == 0) {
    return -1;
  }

  fp = fopen(filepath, "wb");
  if (!fp) {
    return -1;
  }

  // 写入魔数和版本
  fwrite(magic_string, 1, magic_len, fp);
  fwrite(&major_version, 1, 1, fp);
  fwrite(&minor_version, 1, 1, fp);

  // 构造头部信息 - 修正格式
  if (ndim == 1) {
    snprintf(header, sizeof(header), "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu,), }", shape[0]);
  } else {
    // 构造多维数组的shape字符串
    char shape_str[128] = "";
    int  pos            = 0;
    pos += snprintf(shape_str + pos, sizeof(shape_str) - pos, "(");
    for (size_t i = 0; i < ndim; i++) {
      pos += snprintf(shape_str + pos, sizeof(shape_str) - pos, "%zu%s", shape[i], (i < ndim - 1) ? ", " : "");
    }
    pos += snprintf(shape_str + pos, sizeof(shape_str) - pos, ")");

    snprintf(header, sizeof(header), "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);
  }

  // 计算header长度，确保是16的倍数
  header_len       = strlen(header);
  uint16_t padding = 16 - (10 + header_len) % 16;
  if (padding < 16) {
    // 添加空格作为填充
    for (int i = 0; i < padding; i++) {
      strcat(header, " ");
    }
    header_len = strlen(header);
  }

  // 写入头部长度和信息
  fwrite(&header_len, sizeof(header_len), 1, fp);
  fwrite(header, 1, header_len, fp);

  // 写入数据
  size_t total_elements = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_elements *= shape[i];
  }
  fwrite(data, sizeof(float), total_elements, fp);

  fclose(fp);
  return 0;
}

/**
 * @brief 从内存中解析numpy header信息
 * @param data: numpy数据的内存地址
 * @param size: 数据大小
 * @param arr: numpy数组结构体
 * @return 成功返回0，失败返回-1
 */
int parse_npy_header_from_mem(void* data, size_t size, npy_array* arr)
{
  if (!data || !arr || size < 10) {
    return -1;
  }

  const char* ptr = (const char*)data;

  // 检查magic string
  if (memcmp(ptr, "\x93NUMPY", 6) != 0) {
    printf("Invalid numpy magic string\n");
    return -1;
  }
  ptr += 6;

  // 检查版本
  uint8_t major = *ptr++;
  uint8_t minor = *ptr++;
  if (major != 1 || minor != 0) {
    printf("Unsupported numpy format version %d.%d\n", major, minor);
    return -1;
  }

  // 读取header长度
  uint16_t header_len = *(uint16_t*)ptr;
  ptr += 2;

  if (size < (size_t)(10 + header_len)) {
    printf("Data size too small for header\n");
    return -1;
  }

  // 解析header字符串
  char header[256];
  if (header_len >= sizeof(header)) {
    printf("Header too long\n");
    return -1;
  }
  memcpy(header, ptr, header_len);
  header[header_len] = '\0';

  // 查找shape信息
  char* shape_start = strstr(header, "'shape': (");
  if (!shape_start) {
    printf("Cannot find shape in header\n");
    return -1;
  }
  shape_start += 9; // 跳过"'shape': ("

  // 解析shape
  if (parse_shape(shape_start, arr) != 0) {
    printf("Failed to parse shape\n");
    return -1;
  }

  // 查找dtype信息
  char* dtype_start = strstr(header, "'descr': '");
  if (!dtype_start) {
    printf("Cannot find dtype in header\n");
    return -1;
  }
  dtype_start += 10; // 跳过"'descr': '"

  // 解析dtype
  arr->dtype = parse_dtype(dtype_start);

  // 设置数据开始位置
  arr->data_begin    = 10 + header_len;
  arr->raw_data      = data;
  arr->raw_data_size = size;

  return 0;
}