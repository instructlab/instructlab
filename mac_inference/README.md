# Inferencing AI Models on a Mac Laptop

Inference an AI model on a Mac laptop using the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python/) which provides Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Prerequisite

Tested with:

 * Python 3.11.6, 3.11.7
   * CLang distribution of Python: 15.0.0 (xcode)
* MacOS
  * 14.2.1, 14.3.1
  * M1 (Metal/GPU and CPU)

**Note:** The steps shown use [Python venv](https://docs.python.org/3/library/venv.html) for virtual environments. If you have used [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or another tool for Python version management on your laptop, then use the virtual environment with that tool instead. Otherwise, you may have issues with packages installed but modules from that package not found as they are linked to you Python version management tool and not `venv`..

## Converting a Model to GGUF and Quantizing (Optional)

The latest [llama.cpp](https://github.com/ggerganov/llama.cpp) framework requires the model to be converted into [GGUF](https://medium.com/@sandyeep70/ggml-to-gguf-a-leap-in-language-model-file-formats-cd5d3a6058f9) format. [GGUF](https://medium.com/@sandyeep70/ggml-to-gguf-a-leap-in-language-model-file-formats-cd5d3a6058f9) is a quantization technique. [Quantization](https://www.tensorops.ai/post/what-are-quantized-llms) is a technique used to reduce the size of large neural networks, including large language models (LLMs) by modifying the precision of their weights. If you have a model already in GGUF format, you can skip this step.

### Clone the llama.cpp repository

```shell
git clone https://github.com/ggerganov/llama.cpp.git
```

### Set up the virtual environment

```shell
cd llama.cpp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Modify the conversion script

The conversion script has a bug when converting the model.

In `convert-hf-to-gguf.py`, add the following lines (with `+`):

```diff
[...]
def write_tensors(self):
[...]
    self.gguf_writer.add_tensor(new_name, data)
 
+   if new_name == "token_embd.weight":
+       self.gguf_writer.add_tensor("output.weight", data)
+
def write(self):
    self.write_tensors()
[...]
```

### Convert a model to GGUF

The following command converts a Hugging Face model (`safetensors`) to [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format and saves it in your model directory with a `.gguf` extension.

```shell
export MODEL_DIR={model_directory}
python convert-hf-to-gguf.py $MODEL_DIR --outtype f16
```

> Note: This may take about a minute or so.

### Quantize

Optionally, for smaller/faster models with varying loss of quality use a quantized model.

#### Make the llama.cpp binaries

Build binaries like `quantize` etc. for your environment.

```shell
make
```

#### Run quantize command


```shell
./quantize {model_directory}/{f16_gguf_model} <type>
```

For example, the following command converts the f16 GGUF model to a Q4_K_M quantized model and saves it in your model directory with a `<type>.gguf` suffix (e.g. `ggml-model-Q4_K_M.gguf`).

```shell
./quantize $MODEL_DIR/ggml-model-f16.gguf Q4_K_M
```

> Tip: Use `./quantize help` for a list of quantization types with their relative size and output quality along with additional usage parameters.

## Hosting and Inferencing Models

Use the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework to load and run models.

We are using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python/) to provide python bindings to the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework which is C/C++.

> *Note:* From here down, use a new project directory (not the above). There is a [requirements.txt](./requirements.txt) file which will install the Python package dependencies for all the examples that follow in this section. You can use this or the individual requirements file [here](./requirements/). If you would prefer to just install the packages required per example, then follow the manual instructions in the example instead.

### Load and run model using python

Choose your own directory name in place of `_project_dir_`!

```shell
mkdir project_dir
cd project_dir
python3 -m venv venv
source venv/bin/activate
pip install llama-cpp-python==0.2.44
```

Save this code in a .py file or get it [here](./model_load_run.py):

```python
import os
from llama_cpp import Llama

# Get model_path from env
MODEL_FILE=os.getenv("MODEL_FILE")

# Load the model
#   n_gpu_layers=0 to use CPU only
#   n_gpu_layers=-1 for all layers on GPU
#   Specify the number of layers if needed to fit smaller GPUs
model = Llama(model_path=MODEL_FILE, n_gpu_layers=-1)

# Prompt template
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
```

Run using your venv and model file like this:

```shell
source venv/bin/activate
export MODEL_DIR={model_directory}
export MODEL_FILE=$MODEL_DIR/ggml-model-Q4_K_M.gguf
python model_load_run.py
```

## Host model with HTTP server

Choose your own directory name in place of _server_dir_!

```shell
mkdir server_dir
cd server_dir
python3 -m venv venv
source venv/bin/activate
pip install 'llama-cpp-python[server]'
```

Run the server which loads the model and provides an [OpenAI API](https://llama-cpp-python.readthedocs.io/en/latest/server/):

```shell
python3 -m llama_cpp.server --model $MODEL_FILE --api_key "foo" --n_gpu_layers -1
```

**Note:** MODEL_FILE is environment variable for the full pathname of the model file.

**Note:** Exclude `--n_gpu_layers -1` flag if want to run for CPU

### Inference model hosted by the server

Choose your own directory name in place of _client_dir_!

```shell
mkdir client_dir
cd client_dir
python3 -m venv venv
source venv/bin/activate
pip install openai
```

Inference the model by saving this code in a .py file or get it [here](./model_run_from_server.py):

```python
import os
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="foo")

# Get model_path from env
MODEL_FILE_NAME=os.getenv("MODEL_FILE_NAME")

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
messages=[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": usr_prompt}
]

# Inference the model
response = client.chat.completions.create(
  model=MODEL_FILE_NAME,
  # response_format={ "type": "json_object" },
  messages=messages,
  stream=stream_enabled  # Toggle streaming
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
```

Run using your venv and inference the model like this:

```shell
source venv/bin/activate
export MODEL_FILE_NAME=ggml-model-Q4_K_M.gguf
python model_run_from_server.py
```

### Run from vscode

The easiest way to use [vscode](https://code.visualstudio.com) is to start it within the directory of any of the projects you created above. From [Load and Run using Python](#load-and-run-model-using-python) or [Inference model hosted by the server](#inference-model-hosted-by-the-server), you can run the command `code .` and vscode will start for the project.

As you are running it from the project directory, it will pickup the relevant packages installed in the python virtual environment. You can then [run and debug it like any python scripts](https://code.visualstudio.com/docs/python/debugging).

**Note:** You need vscode and Python installed first. Check out [Getting started with Visual Studio Code](https://code.visualstudio.com/docs/introvideos/basics) for more details. 

## Useful Links

* llama.cpp: https://github.com/ggerganov/llama.cpp
* llama-cpp-python: https://github.com/abetlen/llama-cpp-python/
* https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md
* llama-cpp-python OpenAI Compatible Server: https://llama-cpp-python.readthedocs.io/en/latest/server/
* https://www.datacamp.com/tutorial/llama-cpp-tutorial
* https://www.substratus.ai/blog/converting-hf-model-gguf-model/
* Quantize options: https://github.com/ggerganov/llama.cpp/discussions/2094#discussioncomment-6351796
