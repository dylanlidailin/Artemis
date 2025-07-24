import os
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, APIStatusError

# 从环境变量获取 API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("错误：请设置 OPENAI_API_KEY 环境变量。")
    exit()

client = OpenAI(api_key=api_key)

def test_openai_api():
    print("正在尝试连接 OpenAI API...")
    try:
        "Write a poem about chatgpt"
        # 使用一个您有权限访问的模型，例如 "gpt-3.5-turbo" 或 "gpt-4"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 您可以替换为 "gpt-4" 或其他模型
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=20,
            timeout=10.0 # 设置一个超时时间，防止长时间等待
        )

        # 如果成功，打印模型的回复
        print("\n--- API 请求成功！---")
        print(f"模型回复: {response.choices[0].message.content}")
        print(f"Usage: {response.usage}")

    except AuthenticationError:
        print("\n--- 认证失败 ---")
        print("错误：API Key 无效或过期。请检查您的 OPENAI_API_KEY。")
    except APIConnectionError as e:
        print("\n--- 连接错误 ---")
        print(f"错误：无法连接到 OpenAI API。请检查您的网络连接或 API 端点。详细信息: {e}")
    except RateLimitError:
        print("\n--- 速率限制错误 ---")
        print("错误：您已超出 OpenAI API 的速率限制。请稍后重试或检查您的配额。")
    except APIStatusError as e:
        print("\n--- API 状态错误 ---")
        print(f"错误：OpenAI API 返回非 2xx 状态码。状态码: {e.status_code}, 消息: {e.response}")
    except Exception as e:
        print("\n--- 发生未知错误 ---")
        print(f"错误：{e}")

if __name__ == "__main__":
    test_openai_api()