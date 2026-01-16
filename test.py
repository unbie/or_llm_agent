import os
from openai import OpenAI

# 初始化Openai客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 修复点：直接填入 API Key 字符串，或者确保使用正确的 os.environ.get("变量名")
    api_key="b0a7b8f4-397c-4e33-9c5a-12c31120a0f5",
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model="ep-20251202173916-9j664",
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "你好"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("\n----- streaming request -----")
stream = client.chat.completions.create(
    model="ep-20251202173916-9j664",
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "你好"},
    ],
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    print(chunk.choices[0].delta.content, end="")
print()