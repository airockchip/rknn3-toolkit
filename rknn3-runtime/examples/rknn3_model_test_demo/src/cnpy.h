/*
 * @Author: Chifred
 * @Date: 2025-04-27 15:45:47
 * @LastEditors: raul.rao
 * @LastEditTime: 2025-06-29 17:13:25
 * @Editors: Chifred
 * @Description: TODO
 */
#ifndef __CNPY_H__
#define __CNPY_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// numpy数据类型枚举
typedef enum
{
  CNPY_TYPE_FLOAT32,
  CNPY_TYPE_FLOAT64,
  CNPY_TYPE_INT8,
  CNPY_TYPE_INT16,
  CNPY_TYPE_INT32,
  CNPY_TYPE_INT64,
  CNPY_TYPE_UINT8,
  CNPY_TYPE_UINT16,
  CNPY_TYPE_UINT32,
  CNPY_TYPE_UINT64,
  CNPY_TYPE_BOOLEAN,
} cnpy_type;

// numpy数组结构体
typedef struct
{
  void*     raw_data;      // 原始数据指针
  size_t    raw_data_size; // 原始数据大小
  size_t    data_begin;    // 实际数据开始位置
  size_t*   shape;         // 数组形状
  size_t    ndim;          // 维度数
  cnpy_type dtype;         // 数据类型
  bool      fortran_order; // 是否是Fortran顺序
} npy_array;

// 打开numpy文件
int npy_open(const char* filepath, npy_array* arr);

// 关闭numpy数组，释放资源
void npy_close(npy_array* arr);

// 保存float数据到numpy文件
int npy_save_float_buffer_to_file(const char* filepath, float* data, size_t size, size_t* shape, size_t ndim);

int parse_npy_header_from_mem(void* data, size_t size, npy_array* arr);

#endif /* __CNPY_H__ */
