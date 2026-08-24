#pragma once
#include <string>
#include <vector>

enum TokenizerBackendType
{
    TOKENIZER_BACKEND_NONE  = 0,
    TOKENIZER_BACKEND_LLAMA = 1,
};

struct VocabInfo
{
    int vocab_size;
    int special_bos_id[64];
    int special_eos_id[64];
    int n_special_bos_id;
    int n_special_eos_id;
    int linefeed_id;
};

class TokenizerBase;

class Tokenizer {
public:
    /**
     * @brief 构造函数，从分词器文件中提取词表等信息，初始化分词器
     * @param type 分词器后端类型（如Llama等）
     * @param tokenizer_path 分词器文件路径（文件通常为GGUF格式）
     */
    Tokenizer(TokenizerBackendType type, const char* tokenizer_path);

    /**
     * @brief 构造函数，从分词器文件内存中提取词表等信息，初始化分词器
     * @param type 分词器后端类型（如Llama等）
     * @param tokenizer_buffer 分词器文件内存指针
     * @param buffer_size 分词器文件内存的大小
     */
    Tokenizer(TokenizerBackendType type, const void * tokenizer_buffer, size_t buffer_size);

    /**
     * @brief 析构函数，释放分词器相关资源
     */
    ~Tokenizer();

    /**
     * @brief 获取词表信息
     * @param info 用于存储词表信息的结构体指针
     * @return 获取成功返回true，否则返回false
     */
    bool GetVocabInfo(VocabInfo* info);

    /**
     * @brief 将输入文本分词为token序列
     * @param text 输入文本指针
     * @param text_len 输入文本长度
     * @param tokens 输出token数组指针
     * @param n_tokens_max tokens数组的最大容量
     * @return 实际分词得到的token数量，失败返回负值
     */
    int Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);

    /**
     * @brief 将单个token转换为对应的字符串片段（piece）
     * @param token 单个token
     * @return 对应的字符串片段
     */
    std::string TokenToPiece(int32_t token);

    /**
     * @brief 将token数组还原为原始字符串
     * @param tokens token数组指针
     * @param n_tokens token数组长度
     * @return 还原后的字符串
     */
    std::string Decode(int32_t* tokens, int32_t n_tokens);

private:
    TokenizerBase* tokenizerBase = NULL;
    TokenizerBackendType backendType = TOKENIZER_BACKEND_NONE;
};

