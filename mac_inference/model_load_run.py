# Standard
import os

# Third Party
from llama_cpp import Llama

# Get model_path from env
MODEL_FILE = os.getenv("MODEL_FILE")

# Load the model
#   n_gpu_layers=0 to use CPU only
#   n_gpu_layers=-1 for all layers on GPU
#   Specify the number of layers if needed to fit smaller GPUs
model = Llama(model_path=MODEL_FILE, n_gpu_layers=-1)

# Labrador prompt template
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
prompt = "<|system|>\n" + sys_prompt + "\n<|user|>\n" + usr_prompt + "\n<|assistant|>\n"

# Inference the model
result = model(prompt, max_tokens=200, echo=True, stop="<|endoftext|>")

print("\nJSON Output")
print(result)
print("\n\n\nText Output")
final_result = result["choices"][0]["text"].strip()
print(final_result)
print("\n")
