<!--
SPDX-FileCopyrightText: The InstructLab Authors
SPDX-License-Identifier: Apache-2.0
-->

# Performance Numbers

Performance numbers on Mac laptop for a quantized inference model hosted by the [OpenAI server](https://llama-cpp-python.readthedocs.io/en/latest/server/).


## Pre-requisites

Having quantized model hosted by OpenAI server using instructions from [here](../README.md).

Unless specified, the statistics from this document are based on the following configuration:
 * Python version `3.11.7`
 * MacOS
   * `14.0`
   * `M1` (Metal/GPU and CPU)
 * `llama.cpp` version based on short commit hash `4ed8e4f`
 * Sample script provided [here](../model_run_from_server.py) (based on commit hash `dd70e9b`) usually run 3 times 
 * Inference Model - `Q4_K_M` quantized `v3` of `Labrador.13b`

## Statistics from Sample Runs


| Type             | Run # | Time Taken (ms)| Tokens/second |
| ---------------- | ----- | -------------- | ------------- |
| Load Time        | 1     | 1266.11        |               |
|                  | 2     | 1266.11        |               |
|                  | 3     | 1266.11        |               |
| Sample Time      | 1     | 59.81          | 7741.44       |
|                  | 2     | 16.31          | 7788.54       |
|                  | 3     | 21.41          | 7614.69       |
| Prompt Eval Time | 1     | 1265.86        | 191.96        |
|                  | 2     | 0.00           |               |
|                  | 3     | 0.00           |               |
| Eval Time        | 1     | 19147.45       | 24.13         |
|                  | 2     | 5331.27        | 23.82         |
|                  | 3     | 6795.55        | 23.99         |
| Total Time       | 1     | 22385.64       |               |
|                  | 2     | 5724.43        |               |
|                  | 3     | 7302.31        |               |

**Note:** `--n_gpu_layers -1` flag was set and confirmed `offloaded 41/41 layers to GPU` in the server startup logs


