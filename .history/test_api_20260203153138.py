import openai

# 配置API
api_data = dict(
    api_key="3b2262fa-c113-4f64-90db-10ed2659a583",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

client = openai.OpenAI(
    api_key=api_data['api_key'],
    base_url=api_data['base_url']
)

print("Testing API connection...")
print(f"Base URL: {api_data['base_url']}")
print(f"API Key: {api_data['api_key'][:10]}...{api_data['api_key'][-10:]}")

try:
    response = client.chat.completions.create(
        model="ep-20260106214023-k4p8b",
        messages=[
            {"role": "user", "content": "测试：回复'你好'"}
        ],
        temperature=0,
        max_tokens=50,
        stream=False  # 非流式以简化测试
    )
    
    print("\n✅ API连接成功！")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ API连接失败！")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误详情: {e}")
    
    # 尝试打印更多调试信息
    if hasattr(e, '__dict__'):
        print(f"错误属性: {e.__dict__}")
