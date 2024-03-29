happy path for M chips
```ShellSession
# session 1
mkdir workspace-mlx
cd workspace-mlx
python3 -m venv venv
source venv/bin/activate
pip install "git+ssh://git@github.com/instruct-lab/cli.git@mlx"
ilab init

# downloads initial model 
ilab download

ilab serve

# session 2 (also in workspace-mlx)
cp -r taxonomy/compositional_skills/writing/freeform/jokes/puns taxonomy/compositional_skills/writing/freeform/jokes/puns-copy

# generates synthetic data based on provided skill examples 
ilab generate --num-instructions 10

# converts the model to MLX format and trains it on generated, synthetic data
ilab train --iters 10

# runs the trained model against previously unseen data. The same prompts are executed against the model before and after training, to illustrate the progress of the model
ilab test

# converts the trained MLX model back to GGUF format and quantizes it to be served
ilab convert  

# session 1
# CTRL-C to kill the previous ilab serve

# serves up the newly trained, quantized model
ilab serve --model-path ibm-merlinite-7b-mlx-q-fused-pt/ggml-model-Q4_K_M.gguf

# session 2
ilab chat
```

