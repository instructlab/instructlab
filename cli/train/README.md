happy path
```ShellSession
# session 1
mkdir workspace-mlx
cd workspace-mlx
python3 -m venv venv
source venv/bin/activate
pip install "git+ssh://git@github.com/open-labrador/cli.git@mlx#egg=cli[train,convert]"
lab init
lab download
lab serve

# session 2
cp -r taxonomy/compositional_skills/writing/freeform/jokes/puns taxonomy/compositional_skills/writing/freeform/jokes/puns-copy
lab generate --num-instructions 5
mkdir data_puns
cp taxonomy/train_* data_puns/train_gen.jsonl
cp taxonomy/test_* data_puns/test_gen.jsonl
lab train --data-dir data_puns --model-dir ibm/merlinite-7b --remote -q --iters 10
lab convert --model-dir ibm-merlinite-7b-mlx-q --adapter-file ibm-merlinite-7b-mlx-q/adapters.npz -d -q

# session 1
# CTRL-C to kill the previous lab serve
lab serve --model-path ibm-merlinite-7b-mlx-q-fused-pt/ggml-model-Q4_K_M.gguf

# session 2
lab chat
```