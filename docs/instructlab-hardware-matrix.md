# InstructLab system matrix

> [!IMPORTANT]
> The SDG and training time are always dependent on the size of your dataset.

🟢 Supported, regularly tested by CI

🟡 Supported in theory, but not regularly tested by CI

🔴 Not supported

## Apple Macs

|System Profile |CPU |GPU |RAM |Average SDG time |Average training time |CI support |System profile YAML file
|---------------------------|-------|------|----------------|--------|----|-------|-----------------
|M1 Max  |M1 |Integrated graphics |? |? |? |🟡 |[m1_max.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m1/m1_max.yaml)
|M1 Ultra |M1 |Integrated graphics |? |?|? |🟡 |[m1_ultra.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m1/m1_ultra.yaml)
|M2 (Standard) |M2 |Integrated graphics |? |? |? |🟡 |[m2_yaml.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m2/m2.yaml)
|M2 Max |M2 |Integrated graphics |? |? |? |🟡 |[m2_max.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m2/m2_max.yaml)
|M2 Pro |M2 |Integrated graphics |? |? |? |🟡 |[m2_pro.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m2/m2_pro.yaml)
|M2 Ultra |M2 |Integrated graphics |? |? |? |🟡 |[m2_ultra.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m2/m2_ultra.yaml)
|M3 (Standard) |M3 |Integrated graphics |? |? |? |🟡 |[m3.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m3/m3.yaml)
|M3 Max |M3 |Integrated graphics |? |? |? |🟡 |[m3_max.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m3/m3_max.yaml)
|M3 Pro |M3 |Integrated graphics |? |? |? |🟡 |[m3_pro.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/apple/m3/m3_pro.yaml)

## NVIDIA Accelerators

|System Profile |CPU |GPU |RAM |Average SDG time |Average training time |CI support |System profile YAML file
|---------------------------|-------|------|----------------|--------|---|-----------|-----------------
|2x A100 |? |A100 |160 GB |? |? |🟡|[a100_x2.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/a100/a100_x2.yaml)
|4x A100 |? |A100 |320 GB |? |? |🟡 |[a100_x4.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/a100/a100_x4.yaml)
|8x A100 |? |A100 |640 GB |? |? |🟡 |[a100_x4.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/a100/a100_x8.yaml)
|2x H100 |? |H100 |160 GB |? |? |🟡|[h100_x2.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/h100/h100_x2.yaml)
|4x H100 |? |H100 |320 GB |? |? |🟡|[h100_x4.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/h100/h100_x4.yaml)
|8x H100 |? |H100 |640 GB |? |? |🟡|[h100_x4.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/h100/h100_x8.yaml)
|8x L4 |? |L4 |192 GB |? |? |🟡 |[l4_x8.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/l4/l4_x8.yaml)
|4x L4Os |? |L40s |192 GB |? |? |🟢 |[l40s_x4.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/l40s/l40s_x4.yaml)
|8x L4Os |? |L40s |384 GB |? |? |🟡 | [l40s_x8.yaml](https://github.com/instructlab/instructlab/blob/main/src/instructlab/profiles/nvidia/l4/l4_x8.yaml)

## AMD Accelerators

|System Profile |CPU |GPU |RAM |Average SDG time |Average training time |CI Support |System profile YAML file
|---------------------------|-------|------|----------------|--------|-----|-----|-----------------
|:) |? |:) |? |? |? |:) |:)

## Intel Accelerators

|System Profile |CPU |GPU |RAM |Average SDG time |Average training time |CI Support |System profile YAML file
|---------------------------|-------|------|----------------|--------|-----|------|-----------------
|:) |? |:) |:) |? |? |:) |:)