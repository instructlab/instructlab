# InstructLab LLM guide

|Large Language Model (LLM) |Format |Description  |Serving Support |Purpose |Access |Hugging face link
|---------------------------|-------|------|----------------|--------|-------|-----------------
|`Granite-7b-lab`  |Safetensor |Lab fine-tuned Granite model |Linux with GPU  |Default student model  |Manual download| [granite-7b-lab](https://huggingface.co/instructlab/granite-7b-lab)
|`Granite-7b-lab-GGUF` |GGUF |Quantized version of the `granite-7b-lab` model |Mac or Linux |Default chat model |Downloads with default `ilab model download` command  | [granite-7b-lab-GGUF](https://huggingface.co/instructlab/granite-7b-lab-GGUF)
|`Merlinite-7b-lab-GGUF` |GGUF |Quantized version of the `Merlinite-7b-lab` model |Mac or Linux |Teacher model with SDG `simple` pipeline |Downloads with default `ilab model download` command | [merlinite-7b-lab-GGUF](https://huggingface.co/instructlab/merlinite-7b-lab-GGUF)
|`Mistral-7B-Instruct-v0.2-GGUF` |GGUF |Quantized version of the `Mistral-7B-Instruct-v0.2` model |Mac or Linux | Teacher model for SDG `full` pipeline |Downloads with default `ilab model download` command |[Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
|`Prometheus-8x7b-v2.0` |Safetensor |Evaluation model |Linux with GPU |Judge model |Manual download | [prometheus-8x7b-v2.0](https://huggingface.co/prometheus-eval/prometheus-8x7b-v2.0)
|`Merlinite-7b-lab` |Safetensor |LAB fine-tuned `Merlinite-7b` model |Linux with GPU |Student model |Manual download | [merlinite-7b-lab](https://huggingface.co/instructlab/merlinite-7b-lab)

For reference to terms, see [The InstructLab glossary](https://github.com/instructlab/community/blob/main/FAQ.md#glossary).