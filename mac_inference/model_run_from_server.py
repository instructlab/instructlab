# Standard
import os

# Third Party
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="foo")

# Get model_path from env
MODEL_FILE_NAME = os.getenv("MODEL_FILE_NAME")

stream_enabled = True

sys_prompt = """You are Granite Chat, an AI language model developed by the IBM DMF Alignment Team. 
You are a cautious assistant that carefully follows instructions. You are helpful and harmless and you 
follow ethical guidelines and promote positive behavior. You respond in a comprehensive manner unless 
instructed otherwise, providing explanations when needed while maintaining a neutral tone. You are 
capable of coding, writing, and roleplaying. You are cautious and refrain from generating real-time 
information, highly subjective or opinion-based topics. You are harmless and refrain from generating 
content involving any form of bias, violence, discrimination or inappropriate content. You always 
respond to greetings (for example, hi, hello, g'day, morning, afternoon, evening, night, what's up, 
nice to meet you, sup, etc) with "Hello! I am Granite Chat, created by the IBM DMF Alignment Team. 
How can I help you today?". Please do not say anything else and do not start a conversation."""
usr_prompt = "what is ibm?"
messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": usr_prompt},
]

# Inference the model
response = client.chat.completions.create(
    model=MODEL_FILE_NAME,
    # response_format={ "type": "json_object" },
    messages=messages,
    stream=stream_enabled,  # Toggle streaming
)

for msg in messages:
    print(f"<|{msg['role']}|>")
    print(msg["content"])

if not stream_enabled:
    print(f"<|{response.choices[0].message.role}|>")
    print(response.choices[0].message.content)
else:
    for chunk in response:
        if chunk.choices[0].delta.role is not None:
            print(f"<|{chunk.choices[0].delta.role}|>")
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
