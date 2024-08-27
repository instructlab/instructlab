---
author: Kai Xu
date: April 16th, 2024 | v0.13.0
---

# Welcome to Demo for InstructLab CLI

A tool to help you contribute to InstructLab Taxonomy.
<!-- pause -->
It allows you to locally
<!-- pause -->
- Check the behavior of the latest Merlinite-7B model
- Iterate seed examples for synthetic data generation
- Validate the synthetic data via local LoRA training
<!-- pause -->
This demo walks through the functionality step by step.
<!-- pause -->
For more info, see [the CLI repo](https://github.com/instructlab/instructlab).

---

## Step 0: Initial setup

Create workspace and virtual environment

```fish
python3 -m venv workspace/venv
cd workspace
source venv/bin/activate.fish
```

<!-- pause -->
Install CLI

```fish
pip install instructlab[cpu]
```

<!-- pause -->
Initialize workspace

```fish
ilab config init
```

---

## Step 1: Check current model behavior

Download current model

```fish
ilab model download
```

<!-- pause -->
Start chat

```fish
ilab model chat
```

<!-- pause -->
Trying...
> how to initialize a workspace using the CLI of InstructLab?
<!-- pause -->
...but the answer is wrong

---

## Step 2: Write seed examples for SDG

Check current status of taxonomy

```fish
ilab taxonomy diff
```

<!-- pause -->
Check YAML file to add

```fish
cat ../qna.yaml
```

<!-- pause -->
Add file to taxonomy and check updated status of taxonomy

```fish
mkdir -p taxonomy/knowledge/instructlab/cli
cp ../qna.yaml taxonomy/knowledge/instructlab/cli/qna.yaml
ilab taxonomy diff
```

---

## Step 3: Generate synthetic data

Generate 25 synthetic samples

```fish
ilab data generate --num-instructions 25
```

- Here we use 25 for demo purpose only.
- Usually we recommend more samples (e.g. 100).
<!-- pause -->
*It takes a while, feel free to fast-forward if you are watching the recording.*
<!-- pause -->
Check output

```fish
ls
ls generated
```

---

## Step 4: Update model via LoRA training

Perform Q-LoRA for 100 iterations

```fish
ilab model train --iters 100
```

- Here we use 100 for this particular demo.
- The actual iterations needed depends on the complexity of the data.
<!-- pause -->
*It takes a while, feel free to fast-forward if you are watching the recording.*
<!-- pause -->
Check output

```fish
ls
ls instructlab-merlinite-7b-lab-mlx-q
```

---

## Step 5: Check before/after behavior

Run old/new model on test data

```fish
ilab model test
```

---

## Step 6: Interact with updated model

Convert updated model to GGUF

```fish
ilab model convert --model-dir instructlab-merlinite-7b-lab-mlx-q
```

<!-- pause -->
Serve updated model and chat

```fish
ilab model serve --model-path instructlab-merlinite-7b-lab-mlx-q-fused-pt/*-Q4_K_M.gguf &
ilab model chat
```

<!-- pause -->
Trying again...
> how to initialize a workspace using the CLI of InstructLab?

...and the answer is now correct!

---

## Step -1: Make contribution to taxonomy

Commit change and push to fork

```fish
cd taxonomy
git status
git remote add xukai92 https://github.com/instructlab/taxonomy.git
git add knowledge/instructlab/cli
git commit -sm "feat(knowledge): InstructLab CLI usage"
git checkout -b demo
git push --set-upstream xukai92 demo
```

<!-- pause -->
*Now, you can go creating a pull request on the taxonomy repository!*

--> [instructlab/taxonomy](https://github.com/instructlab/taxonomy)
<!-- pause -->
For more info, see [the CLI repo](https://github.com/instructlab/instuctlab).
