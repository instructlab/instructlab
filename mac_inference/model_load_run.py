# Standard
import os

# Third Party
from llama_cpp import Llama

# Local
from . import utils

# Get model_path from env
MODEL_FILE = os.getenv("MODEL_FILE")

# Load the model
#   n_gpu_layers=0 to use CPU only
#   n_gpu_layers=-1 for all layers on GPU
#   Specify the number of layers if needed to fit smaller GPUs
model = Llama(model_path=MODEL_FILE, n_gpu_layers=-1)

# System prompt
usr_prompt = "what is ibm?"
prompt = (
    "<|system|>\n"
    + utils.get_sysprompt()
    + "\n<|user|>\n"
    + usr_prompt
    + "\n<|assistant|>\n"
)

# Inference the model
result = model(prompt, max_tokens=200, echo=True, stop="<|endoftext|>")

print("\nJSON Output")
print(result)
print("\n\n\nText Output")
final_result = result["choices"][0]["text"].strip()
print(final_result)
print("\n")
