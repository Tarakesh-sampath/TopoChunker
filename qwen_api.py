# from openai import OpenAI
# import os

# client = OpenAI(
#     # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
#     api_key="sk-ed30d820ec414bba8b78fd0b03de3693",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# messages = [{"role": "user", "content": "你是谁"}]
# completion = client.chat.completions.create(
#     model="qwen-plus",  # 您可以按需更换为其它深度思考模型
#     messages=messages,
#     extra_body={"enable_thinking": True},
#     stream=True
# )
# is_answering = False  # 是否进入回复阶段
# print("\n" + "=" * 20 + "思考过程" + "=" * 20)
# for chunk in completion:
#     delta = chunk.choices[0].delta
#     if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
#         if not is_answering:
#             print(delta.reasoning_content, end="", flush=True)
#     if hasattr(delta, "content") and delta.content:
#         if not is_answering:
#             print("\n" + "=" * 20 + "完整回复" + "=" * 20)
#             is_answering = True
#         print(delta.content, end="", flush=True)

# import os
# from openai import OpenAI

# try:
#     client = OpenAI(
#         # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
#         api_key="sk-ed30d820ec414bba8b78fd0b03de3693",
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )

#     completion = client.chat.completions.create(
#         model="qwen-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
#         messages=[
#             {'role': 'system', 'content': 'You are a helpful assistant.'},
#             {'role': 'user', 'content': '你是谁？'}
#         ]
#     )
#     print(completion.choices[0].message.content)
# except Exception as e:
#     print(f"错误信息：{e}")
#     print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")

# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-c6940cb5b0064b53af84f8ffa67cc401",
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)