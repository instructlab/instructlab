import os

def train(data_dir, model_dir, remote, quantize):
    """
    Takes synthetic data generated locally with `lab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """

    is_macos = True # TODO detect OS
    if is_macos:
        # NOTE we can skip this if we have a way ship MLX
        # TODO convert the model from PyTorch to MLX
        # PyTorch safetensors to MLX safetensors
        model_dir_mlx = f"{model_dir}-mlx"
        model_dir_mlx_quantized = f"{model_dir}-mlx-q"

        dest_model_dir = ""
        quantize_arg = ""
        local_arg = ""
        if quantize:
            dest_model_dir = model_dir_mlx_quantized
            quantize_arg =  "-q"
        else:
            dest_model_dir = model_dir_mlx

        if not remote:
            local_arg = "--local"

        script = os.path.join(os.getcwd(), "train/lora-mlx/convert.py")
        cmd = f"{script}  --hf-path {model_dir} --mlx-path {dest_model_dir} {quantize_arg} {local_arg}"
         # python convert.py --hf-path ibm-merlinite-7b --mlx-path ibm-merlinite-7b-mlx
        os.system('python {}'.format(cmd))


        adapter_file_path = f"{dest_model_dir}/adapters.npz"
        script = os.path.join(os.getcwd(), "train/lora-mlx/lora.py")
        # TODO train the model with LoRA
        # python lora.py --model {{model_dir_mlx}} --train --data data_puns_shiv --adapter-file {{model_dir_mlx}}/adapters.npz --iters 300 --save-every 10 --steps-per-eval 10
        # exact command:
        # python lora.py --model ibm-merlinite-7b-mlx --train --data data_puns_shiv --adapter-file ibm-merlinite-7b-mlx/adapters.npz --iters 100 --save-every 10 --steps-per-eval 10
        cmd = f"{script} --model {dest_model_dir} --train --data {data_dir} --adapter-file {adapter_file_path} --iters 300 --save-every 10 --steps-per-eval 10"
        os.system('python {}'.format(cmd))

        # TODO copy some downloaded files from the PyTorch model folder
        # Seems to be not a problem if working with a remote download with convert.py
        # just copy-files ibm-merlinite-7b-mlx

    else:
        click.secho(
            f"`lab train` is only implemented for macOS with M-series chips",
            fg="red",
        )

    #   Can this target a directory or does it overwrite the model on the --model directory?
    pass




def test(model_dir, adapter_file):
    script = os.path.join(os.getcwd(), "train/lora-mlx/lora.py")
    cmd = f"{script} --model {model_dir} --adapter-file {adapter_file} --max-tokens 100 --prompt \"<|system|>\nYou are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{{prompt}}\n<|assistant|>\n\""
    # _generate-no-lora model prompt:
    #     python lora.py --model {{model}} --no-adapter --max-tokens 100 --prompt "<|system|>\nYou are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{{prompt}}\n<|assistant|>\n"
    # _generate model adapter prompt:
    #     python lora.py --model {{model}} --adapter-file {{model}}/{{adapter}} --max-tokens 100 --prompt "<|system|>\nYou are Labrador, an AI language model developed by IBM DMF (Data Model Factory) Alignment Team. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{{prompt}}\n<|assistant|>\n"
    os.system('python {}'.format(cmd))

def convert(model_dir, adapter_file, quantized=False):
    
    """
    TODO
    """
    model_dir_mlx = f"{model_dir}-mlx"
    model_dir_mlx_quantized = f"{model_dir}-mlx-q"

    dequantize_arg = ""
    source_model_dir = model_dir
    if quantized:
        dequantize_arg = " -d "

    model_dir_fused= f"{source_model_dir}-fused"
    
    script = os.path.join(os.getcwd(), "train/lora-mlx/fuse.py")
    # cmd = cwd + " --model " + source_model_dir + " --save-path " + model_dir_fused + " --adapter-file " + adapter_file_path + dequantize_arg
    cmd = f"{script} --model {source_model_dir} --save-path {model_dir_fused} --adapter-file {adapter_file} {dequantize_arg}"
    # this combines adapter with the original model to produce the updated model
    # python fuse.py --model {{model}} --save-path {{model}}-fused --adapter-file {{model}}/adapters-100.npz
    # just copy-files {{model}}-fused
    os.system('python {}'.format(cmd))

    model_dir_fused_pt= f"{model_dir_fused}-pt"

    script = os.path.join(os.getcwd(), "train/lora-mlx/convert.py ")
    cmd = f"{script} --hf-path { model_dir_fused} --mlx-path {model_dir_fused_pt} --local --to-pt"
    # this converts MLX to PyTorch
    # python convert.py --hf-path {{model}} --local --mlx-path {{model}}-pt --to-pt
    # just copy-files {{model}}-pt
    os.system('{} {}'.format('python', cmd))


    script = os.path.join(os.getcwd(), "llamacpp/llamacpp_convert_to_gguf.py")
    cmd = f"{script} { model_dir_fused_pt} --pad-vocab"
    # use llama.cpp to convert back to GGUF
    # python $HOME/src/open-labrador/llama.cpp/convert.py {{model_dir}} --pad-vocab
    # TO DO: fix this to execute function instead
    os.system('{} {}'.format('python', cmd))

    # quantize 4-bi GGUF (optional)
    # $HOME/src/open-labrador/llama.cpp/quantize {{model_dir}}/ggml-model-f16.gguf {{model_dir}}/ggml-model-Q4_K_M.gguf Q4_K_M
    # TO DO: fix this to execute function instead
    if quantized:
        gguf_model_dir = f"{model_dir_fused_pt}/ggml-model-f16.gguf" 
        gguf_model_q_dir = f"{model_dir_fused_pt}/ggml-model-Q4_K_M.gguf"
        script = os.path.join(os.getcwd(), "llamacpp/quantize")
        cmd = f"{script} {gguf_model_dir} {gguf_model_q_dir} Q4_K_M"
        os.system('{}'.format(cmd))