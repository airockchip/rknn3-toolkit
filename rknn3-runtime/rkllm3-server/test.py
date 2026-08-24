import openai
import readline     # 避免input函数在删除中文字符出现半字符异常, 不能去掉

def chat_with_ai(client, prompt, cache_prompt=True):
    response = client.chat.completions.create(
        model="",
        messages=prompt,
        stream=True,                        # 流模式
        extra_body={
            "n_keep": 0,                    # 超出context时, 保留前n_keep个kvcache的状态不被覆盖
            "cache_prompt": cache_prompt,   # 对应于runtime的keep_history, 为False时则回清空kvcache, 相当于重置会话
            "id_slot": 0,                   # 对应slot, 单应用时填0即可
            "n_predict": 64                 # 一次会话最多返回n_predict个token, 对应于runtime里的max_new_tokens
        }
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            delta.content = delta.content.replace(f'\n', '<br/>')
            # yield f"data: {delta.content}\n\n"
        if chunk.choices[0].finish_reason == "stop":
            break
        print(delta.content, end='', flush=True)
    print('')

if __name__ == "__main__":
    client = openai.OpenAI(
        # base_url="http://172.16.10.20:8080/v1",
        # base_url="http://172.16.10.172:8080/v1",
        base_url="http://172.16.10.106:8080/v1",
        api_key = "sk-no-key-required"
    )
    
    print("AI助手已启动，输入'退出'或'quit'结束对话。")

    first = True
    while True:
        user_input = input("你: ")
        if not user_input:
            continue

        # 检查是否退出
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("对话结束。")
            break

        # 重置会话
        if user_input.lower() in ['reset']:
            first = True
            continue

        prompt = []

        # 会话开始时是否添加system的prompt, 也可以注释掉不添加
        # if first:
        #     prompt.append({"role": "system", "content": "You are a helpful assistant."})

        prompt.append({"role": "user", "content": user_input})

        # 获取AI回复
        print("AI: ", end='', flush=True)
        chat_with_ai(client, prompt, cache_prompt=not first)
        first = False