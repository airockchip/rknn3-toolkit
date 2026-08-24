import base64
from openai import OpenAI

client = OpenAI(
    # base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    base_url="http://172.16.10.46:8080/v1",
    api_key = "sk-no-key-required"
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "demo.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
                { "type": "text", "text": "请描述一下图片?" },
            ],
        }
    ],
    stream=True,
    extra_body={
        "n_keep": 0,
        "cache_prompt": False,
        "id_slot": 0,
        "n_predict": 256
    }
)

for chunk in completion:
    delta = chunk.choices[0].delta
    if delta.content:
        delta.content = delta.content.replace(f'\n', '<br/>')
        # yield f"data: {delta.content}\n\n"
    if chunk.choices[0].finish_reason == "stop":
        break
    print(delta.content, end='', flush=True)
print('')
