# RKNN3 模型加密使用方法

RKNN SDK 支持离线加密 RKNN 模型，并在运行时进行解密。

> 模型加密只是模型保护的一部分，无法完全杜绝被破解的风险，建议用户尽量避免模型被第三方直接接触，并在系统层面做好整体防护。

## 安全特性

RKNN3 采用 **混合加密架构**，确保模型安全：

| 特性 | 说明 |
|------|------|
| **双层加密** | AES-256-CTR 加密模型文件 + RSA-OAEP 保护密钥 |
| **密钥安全** | 用户密钥由 Rockchip 公钥加密，只有芯片内置私钥可解密 |
| **链路安全** | 权重文件以密文形式传输，PCIe/USB 链路不暴露明文 |
| **内存安全** | 解密后的密钥仅在内存中短暂存在，使用后立即清除 |

## 1. 加密模型和权重文件

> 公钥由 Rockchip 提供（`rk_public_key.pem`）

```bash
# 使用公钥加密模型和权重
python3 rknn3_model_encrypt.py \
    --model model.rknn \
    --weight model.weight \
    --rk-pubkey rk_public_key.pem \
    --output-dir ./encrypted

# 输出：
# ./encrypted/model.rknn.enc      - 加密后的模型
# ./encrypted/model.weight.enc    - 加密后的权重
# ./encrypted/user_key_iv.enc     - RSA 加密的密钥信封
```

## 2. 部署到开发板

将以下文件拷贝到开发板：
- `model.rknn.enc`
- `model.weight.enc`
- `user_key_iv.enc`

## 3. 板端 API 调用

```c
// 1. 设置解密密钥
rknn3_set_decrypt_key_from_path(ctx, "user_key_iv.enc");

// 2. 加载加密模型（自动解密）
rknn3_load_model_from_path(ctx, "model.rknn.enc", "model.weight.enc");
```

> 具体的使用示例可见 `rknn3_model_test_demo`及`rknn3_session_test_demo` 目录下的 C++ 代码。

## 文件格式

| 文件 | 格式 | 说明 |
|------|------|------|
| `.rknn.enc` / `.weight.enc` | RKCE | AES-256-CTR 加密 |
| `user_key_iv.enc` | RKKE | RSA-OAEP 加密的 AES 密钥 |