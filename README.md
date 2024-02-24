# Labrador 🐶 command-line interface `cli`

Labrador 🐶 is a novel synthetic data-based alignment tuning method for Large 
Language Models (LLMs.) The "**lab**" in **Lab**rador 🐶 stands for **L**arge-
scale **A**lignment for Chat **B**ots.

This command-line interface for Labrador 🐶 will allow you to create models tuned 
with your data using the Labrador 🐶 method on your laptop or workstation.

*This is currently a tool that **requires an M1/M2/M3 Mac** to use; we anticipate 
future support for 🐧 Linux and other operating systems as well as for 
💻 additional hardware.*

## Contents:
* [Getting `cli`](#TODO)
* [How to use `cli`](#TODO)
* [How to convert and quantize a model (Optional)](#TODO)



# Getting `cli`

## 📋 Requirements

- 🐍 Python 3.9 or later (CLang dsitribution of Python: 15.0.0 from xcode)
- 🍎 macOS (14.x with an M1/M2/M3 Metal/GPU) 
- 📦 A quantized model in GGUF format (or read our [guide](#TODO) on to convert 
models to GGUF format and quantize them.)
  
🗒️ **Note:** The steps below use [Python venv](https://docs.python.org/3/library/
venv.html) for virtual environments. If you have used [pyenv](https://github.
com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) 
or another tool for Python version management on your laptop, then use the 
virtual environment with that tool instead. Otherwise, you may have issues with 
packages installed but modules from that package not found as they are linked 
to your Python version management tool and not `venv`.

## 🧰 Installation

`cli` will be available via `pip install lab-cli` in the future. At this time, 
you will need to run `cli` from source:

```ShellSession
git clone https://github.com/open-labrador/cli.git
cd cli
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Running `cli`

Always be sure to activate the venv in your working shell:

```ShellSession
source venv/bin/activate
```

Then, run `cli` as follows:

```ShellSession
python -m cli
```

# How to use `cli`

First, you will need a properly-formatted **[example dataset](#TODO)** to tune 
the model with. Once you have that, 
using the Labrador 🐶 method involves a number of steps, supported by various 
commands:

## 🏗️ 1. Initial setup
- Initialize a local environment to use Labrador 🐶 via the **init**
command:

  `python -m cli init`

- Download the model to train using the **download** command:

  `python -m cli download {URL to gguf-format model}`

  🚧 **Under construction:** This command isn´t ready yet! 😅 Pop over to our 
[model download guide](#TODO) for a set of instructions on how to do this 
manually.

## 🧑‍🏫 2. Model training
- Generate a synthetic dataset to enhance your example data set using the 
**generate** command:

  `python -m cli generate {path to root directory location of dataset}`

- Train the model on your synthetic data-enhanced dataset using **train**:

  `python -m cli train {local path to gguf-format model} {path to root directory 
location of dataset}` 

## 👩🏽‍🔬 3. Testing the fine-tuned model
- Serve the fine-tuned model locally via the **serve** command using the 
[llama.cpp framework](#TODO) and [llama-cpp-python](#TODO) (which provides 
Python bindings for llama.cpp):

  `python -m cli serve {local path to fine-tuned model}`

  🚧 **Under construction:** This command isn´t ready yet! 😅 Pop over to our 
  [model servingguide](#TODO) for a set of instructions on how to do this 
  manually.
- Try the fine-tuned model out live using a chat interface, and see if the 
results are better than the untrained version of the model with **chat**:

  `python -m cli chat`

  📋 **Note:** We have [a detailed guide](#TODO) on using the **chat** command.
- Run tests against the model via the **test** command:

  `python -m cli test`

## 🎁 4. Submit your dataset!

Of course the final step is, if you've improved the model, to share your new 
dataset by submitting it! You'll submit it via a pull-request process, which 
is documented in the [taxonomy respository](#TODO).


# Converting a Model to GGUF and Quantizing (Optional)

The latest [llama.cpp](https://github.com/ggerganov/llama.cpp) framework 
requires the model to be converted into [GGUF](https://medium.com/@sandyeep70/
ggml-to-gguf-a-leap-in-language-model-file-formats-cd5d3a6058f9) format. [GGUF]
(https://medium.com/@sandyeep70/ggml-to-gguf-a-leap-in-language-model-file-
formats-cd5d3a6058f9) is a quantization technique. [Quantization]
(https://www.tensorops.ai/post/what-are-quantized-llms) is a technique used to 
reduce the size of large neural networks, including large language models 
(LLMs) by modifying the precision of their weights. If you have a model already
 in GGUF format, you can skip this step.

## Clone the llama.cpp repo

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

The conversion script has a bug when converting the Labrador 🐶 model.

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

The following command converts a Hugging Face model (safetensors) to [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format and saves it in your model directory with a `.gguf` extension.

```shell
export MODEL_DIR={model_directory}
python convert-hf-to-gguf.py $MODEL_DIR --outtype f16
```

> Note: This may take about a minute or so.

## Quantize

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

For example, the following command converts the f16 GGUF model to a Q4_K_M quantized model and saves it in your model directory with a `<type>.gguf` suffix (e.g. ggml-model-Q4_K_M.gguf).

```shell
./quantize $MODEL_DIR/ggml-model-f16.gguf Q4_K_M
```

> Tip: Use `./quantize help` for a list of quantization types with their relative size and output quality along with additional usage parameters.


