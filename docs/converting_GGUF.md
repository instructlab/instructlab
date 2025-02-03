# Optional: Converting a Model to GGUF and Quantizing

The latest [llama.cpp](https://github.com/ggerganov/llama.cpp) framework
requires the model to be converted into [GGUF](https://medium.com/@sandyeep70/ggml-to-gguf-a-leap-in-language-model-file-formats-cd5d3a6058f9)
format. [GGUF](https://medium.com/@sandyeep70/ggml-to-gguf-a-leap-in-language-model-file-formats-cd5d3a6058f9)
is a quantization technique. [Quantization](https://www.tensorops.ai/post/what-are-quantized-llms)
is a technique used to reduce the size of large neural networks, including large
language models (LLMs) by modifying the precision of their weights. If you have a
model already in GGUF format, you can skip this step.

## Clone the llama.cpp repository

```shell
git clone https://github.com/ggerganov/llama.cpp.git
```

## Set up the virtual environment

```shell
cd llama.cpp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Modify the conversion script

The conversion script has a bug when converting the InstructLab 🥼 model.

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

## Convert a model to GGUF

The following command converts a Hugging Face model (`safetensors`) to [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
format and saves it in your model directory with a `.gguf` extension.

```shell
export MODEL_DIR={model_directory}
python convert-hf-to-gguf.py $MODEL_DIR --outtype f16
```

> Note: This may take about a minute or so.

## Quantize

Optionally, for smaller/faster models with varying loss of quality use a
quantized model.

### Make the llama.cpp binaries

Build binaries like `quantize` etc. for your environment.

```shell
make
```

#### Run quantize command

```shell
./quantize {model_directory}/{f16_gguf_model} <type>
```

For example, the following command converts the f16 GGUF model to a Q4_K_M
quantized model and saves it in your model directory with a `<type>.gguf`
suffix (e.g. `ggml-model-Q4_K_M.gguf`).

```shell
./quantize $MODEL_DIR/ggml-model-f16.gguf Q4_K_M
```

> Tip: Use `./quantize help` for a list of quantization types with their
> relative size and output quality along with additional usage parameters.
