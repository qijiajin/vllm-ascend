# _npu_fused_infer_attention_sink_metadata 算子迁移方案

> 源仓库: `omni-ops`
> 目标仓库: `vllm-ascend` (路径: `vllm-ascend/csrc`)
> 参考实现: `csrc/attention/sparse_attn_sharedkv_metadata/`(同类 metadata 算子)

## 1. 背景与目标

`torch.ops.custom._npu_fused_infer_attention_sink_metadata` 是 `npu_fused_infer_attention_sink`(带 sink token 的融合推理 Attention)算子的前置 metadata 算子,负责在 AICPU 上计算分核 tiling 信息(每个 AICore/AIVector 的任务起止点、FlashDecode 归约信息等),输出一个固定大小(`FIASINK_META_SIZE = 1024`)的 `uint32` 张量,供后续 FIA-Sink 算子消费。

本方案将该算子从 `omni-ops` 迁移到 `vllm-ascend/csrc`,使其遵循 vllm-ascend 的算子组织规范(参考 `sparse_attn_sharedkv_metadata`),并接入 `torch.ops._C_ascend` 命名空间与 `build_aclnn.sh` 构建流水线。

## 2. 源算子分析(omni-ops)

### 2.1 源码位置

| 层 | 路径 |
|---|---|
| CANN 算子(op_api/op_graph/op_host/op_kernel_aicpu) | `omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink_metadata/` |
| 共享 metadata 头(被 infershape 与 aicpu 引用) | `omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink/op_kernel/ai_infra_fused_infer_attention_sink_metadata.h` |
| Torch 绑定(csrc) | `omni-ops/inference/ascendc/torch_ops_extension/omni_custom_ops/ops_transformer/attention/fused_infer_attention_sink_metadata/csrc/npu_fused_infer_attention_sink_metadata.cpp` |
| Torchair GE converter(图模式) | `omni-ops/inference/ascendc/torch_ops_extension/omni_custom_ops/ops_transformer/attention/fused_infer_attention_sink_metadata/converter/npu_fused_infer_attention_sink_metadata.py` |

### 2.2 算子签名(omni-ops)

```
torch.ops.custom._npu_fused_infer_attention_sink_metadata(
    num_heads_q, num_heads_kv, head_dim_qk, head_dim_v, *,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    batch_size=0, sparse_mode=0, pre_tokens=2147483647, next_tokens=2147483647,
    input_layout="TND", input_layout_kv="TND",
    sink_num=0, k_sink_num=0, batch_invariant=False,
    rope_head_dim=0, block_size=0,
    aic_core_num=24, aiv_core_num=48
) -> Tensor   # shape=[1024], dtype=uint32/int32
```

### 2.3 两套 ACLNN 入口

源算子提供两个 ACLNN 入口,通过 `batch_invariant` 分支选择:

- `aclnnAiInfraFusedInferAttentionSinkMetadata`(V1): 不含 `batch_invariant` 形参
- `aclnnAiInfraFusedInferAttentionSinkMetadataV2`(V2): 含 `batch_invariant` 形参

两者共用同一个 l0 算子 `AiInfraFusedInferAttentionSinkMetadata`(`l0_*.cpp` 中 `ADD_TO_LAUNCHER_LIST_AICPU`),并共用同一个 AICPU kernel(`REGISTER_CPU_KERNEL`)。

### 2.4 依赖说明

- `infershape.cpp` 与 `aicpu.cpp` 均 `#include` 了主 FIA-Sink 算子目录下的 `ai_infra_fused_infer_attention_sink_metadata.h`(定义 `FIASINK_META_SIZE`、`FiaSinkMetaData` 结构、各 metadata 索引常量)。**该头文件必须随迁移**。
- AICPU kernel `kernelSo` = `libtransformer_aicpu_kernels.so`(见 `*_aicpu.json`)。

## 3. 目标结构(vllm-ascend/csrc)

遵循 `sparse_attn_sharedkv_metadata` 目录布局,在 `vllm-ascend/csrc/attention/` 下新建算子目录:

```
vllm-ascend/csrc/attention/ai_infra_fused_infer_attention_sink_metadata/
├── op_api/
│   ├── aclnn_ai_infra_fused_infer_attention_sink_metadata.h        # V1 ACLNN 头
│   ├── aclnn_ai_infra_fused_infer_attention_sink_metadata.cpp      # V1 ACLNN 实现
│   ├── aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.h     # V2 ACLNN 头(含 batch_invariant)
│   ├── aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.cpp   # V2 ACLNN 实现
│   ├── l0_ai_infra_fused_infer_attention_sink_metadata.h           # l0 算子声明
│   └── l0_ai_infra_fused_infer_attention_sink_metadata.cpp         # l0 算子(AICPU launcher)
├── op_graph/
│   └── ai_infra_fused_infer_attention_sink_metadata_proto.h        # REG_OP 算子原型注册
├── op_host/
│   └── ai_infra_fused_infer_attention_sink_metadata_infershape.cpp # InferShape/InferDataType
├── op_kernel_aicpu/
│   ├── ai_infra_fused_infer_attention_sink_metadata_aicpu.h        # AICPU kernel 类定义
│   ├── ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp      # AICPU kernel 实现(分核算法)
│   ├── ai_infra_fused_infer_attention_sink_metadata_aicpu.json     # kernel so 配置
│   └── CMakeLists.txt
├── CMakeLists.txt
└── README.md
```

共享 metadata 头迁移至本算子目录内(避免跨算子目录引用):

```
op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata.h   # 由 FIA-Sink op_kernel 迁入
```

`infershape.cpp` 与 `aicpu.cpp` 的 `#include` 路径相应改为本目录内引用。

## 4. 文件级迁移映射

| 源文件(omni-ops) | 目标文件(vllm-ascend/csrc/attention/ai_infra_fused_infer_attention_sink_metadata/) | 改动要点 |
|---|---|---|
| `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata.h` | `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata.h` | 版权年份更新;内容基本不变 |
| `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata.cpp` | `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata.cpp` | `#include "l0_..."` 路径;namespace 由匿名→保持;`AiInfraFusedInferAttentionSinkMetadatal0op::` 调用保留 |
| `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.h` | `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.h` | 同上 |
| `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.cpp` | `op_api/aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.cpp` | 同上 |
| `op_api/l0_ai_infra_fused_infer_attention_sink_metadata.h` | `op_api/l0_ai_infra_fused_infer_attention_sink_metadata.h` | namespace `AiInfraFusedInferAttentionSinkMetadatal0op` 保留 |
| `op_api/l0_ai_infra_fused_infer_attention_sink_metadata.cpp` | `op_api/l0_ai_infra_fused_infer_attention_sink_metadata.cpp` | 内容直接迁移 |
| `op_graph/ai_infra_fused_infer_attention_sink_metadata_proto.h` | `op_graph/ai_infra_fused_infer_attention_sink_metadata_proto.h` | `REG_OP` 内容直接迁移 |
| `op_host/ai_infra_fused_infer_attention_sink_metadata_infershape.cpp` | `op_host/ai_infra_fused_infer_attention_sink_metadata_infershape.cpp` | `#include` 改为本目录 `../op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata.h` |
| `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.h` | `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.h` | 内容直接迁移 |
| `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp` | `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp` | `#include` 路径调整:`../../ai_infra_fused_infer_attention_sink/op_kernel/...` → `./ai_infra_fused_infer_attention_sink_metadata.h`(本目录) |
| `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.json` | `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.json` | `kernelSo` 保持 `libtransformer_aicpu_kernels.so` |
| (新) `ai_infra_fused_infer_attention_sink/op_kernel/ai_infra_fused_infer_attention_sink_metadata.h` | `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata.h` | 共享 metadata 头迁入本目录 |

### 4.1 CMakeLists.txt(新增)

`op_kernel_aicpu/CMakeLists.txt`(参考 `kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/CMakeLists.txt`):

```cmake
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolchain/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  endif()

  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  message(STATUS "[ai_infra_fused_infer_attention_sink_metadata] Found aicpu sources: ${AICPU_SRC}, ascend dir: ${ASCEND_DIR}, ophost name: ${OPHOST_NAME}")

  add_aicpu_cust_kernel_modules(ai_infra_fused_infer_attention_sink_metadata ${AICPU_SRC} ${JSON_FILE})
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    AddAicpuOpTestCase(ai_infra_fused_infer_attention_sink_metadata)
endif()
```

顶层 `CMakeLists.txt`(参考 `sparse_attn_sharedkv_metadata/CMakeLists.txt`):

```cmake
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License")
add_modules_sources_aicpu(DEPENDENCIES ai_infra_fused_infer_attention_sink_metadata)
```

## 5. Torch 绑定迁移

### 5.1 omni-ops 绑定方式(不迁移)

omni-ops 使用独立 `csrc/*.cpp` + `converter/*.py`(torchair `register_fx_node_ge_converter`)分别处理 eager 与图模式,并通过 `ops_def_registration.cpp` 注册到 `torch.ops.custom`。**vllm-ascend 不采用此模式**。

### 5.2 vllm-ascend 绑定方式(目标)

参考 `sparse_attn_sharedkv_metadata`,vllm-ascend 将绑定内联到 `csrc/torch_binding.cpp`(NPU 实现)与 `csrc/torch_binding_meta.cpp`(Meta 实现),通过 `EXEC_NPU_CMD` 宏调用 ACLNN,不再迁移 torchair GE converter(图模式由 aclgraph 覆盖)。

#### 5.2.1 NPU 实现(`csrc/torch_binding.cpp`)

需新增两段:

**(a) 实现函数**(放在 `npu_sparse_attn_sharedkv_metadata_npu` 附近,文件中段 namespace `vllm_ascend` 内):

```cpp
at::Tensor npu_fused_infer_attention_sink_metadata_npu(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim_qk,
    int64_t head_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sink_num,
    int64_t k_sink_num,
    bool batch_invariant,
    int64_t rope_head_dim,
    int64_t block_size,
    int64_t aic_core_num,
    int64_t aiv_core_num,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths.has_value()) {
        output_device = actual_seq_lengths.value().device();
    } else if (actual_seq_lengths_kv.has_value()) {
        output_device = actual_seq_lengths_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE},
                                     torch::dtype(torch::kInt32).device(output_device));

    auto actual_seq_lengths_val = get_valid_tensor(actual_seq_lengths, output_device);
    auto actual_seq_lengths_kv_val = get_valid_tensor(actual_seq_lengths_kv, output_device);

    std::string input_layout_str = std::string(input_layout);
    std::string input_layout_kv_str = std::string(input_layout_kv);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());

    if (batch_invariant) {
        EXEC_NPU_CMD(aclnnAiInfraFusedInferAttentionSinkMetadataV2,
                     actual_seq_lengths_val, actual_seq_lengths_kv_val,
                     num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
                     batch_size, sparse_mode, pre_tokens, next_tokens,
                     input_layout_ptr, input_layout_kv_ptr,
                     sink_num, k_sink_num, batch_invariant,
                     rope_head_dim, block_size,
                     aic_core_num, aiv_core_num, output);
    } else {
        EXEC_NPU_CMD(aclnnAiInfraFusedInferAttentionSinkMetadata,
                     actual_seq_lengths_val, actual_seq_lengths_kv_val,
                     num_heads_q, num_heads_kv, head_dim_qk, head_dim_v,
                     batch_size, sparse_mode, pre_tokens, next_tokens,
                     input_layout_ptr, input_layout_kv_ptr,
                     sink_num, k_sink_num, rope_head_dim, block_size,
                     aic_core_num, aiv_core_num, output);
    }
    return output;
}
```

> 注:`get_valid_tensor` lambda 已存在于 `torch_binding.cpp` 1167 行附近,可直接复用。

**(b) 算子 def + impl 注册**(放在 `torch_binding.cpp` 末段 `init` block,参照 `npu_sparse_attn_sharedkv_metadata` 的 `ops.def/ops.impl` 写法):

```cpp
    ops.def(
        "npu_fused_infer_attention_sink_metadata("
            "int num_heads_q, "
            "int num_heads_kv, "
            "int head_dim_qk, "
            "int head_dim_v, "
            "Tensor? actual_seq_lengths=None, "
            "Tensor? actual_seq_lengths_kv=None, "
            "int batch_size=0, "
            "int sparse_mode=0, "
            "int pre_tokens=2147483647, "
            "int next_tokens=2147483647, "
            "str input_layout='TND', "
            "str input_layout_kv='TND', "
            "int sink_num=0, "
            "int k_sink_num=0, "
            "bool batch_invariant=False, "
            "int rope_head_dim=0, "
            "int block_size=0, "
            "int aic_core_num=24, "
            "int aiv_core_num=48, "
            "str device='npu'"
        ") -> Tensor"
    );
    ops.impl("npu_fused_infer_attention_sink_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_fused_infer_attention_sink_metadata_npu);
```

#### 5.2.2 Meta 实现(`csrc/torch_binding_meta.cpp`)

```cpp
at::Tensor npu_fused_infer_attention_sink_metadata_meta(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim_qk,
    int64_t head_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sink_num,
    int64_t k_sink_num,
    bool batch_invariant,
    int64_t rope_head_dim,
    int64_t block_size,
    int64_t aic_core_num,
    int64_t aiv_core_num,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Tensor output;
    if (actual_seq_lengths.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(actual_seq_lengths.value().device()));
    } else if (actual_seq_lengths_kv.has_value()) {
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(actual_seq_lengths_kv.value().device()));
    } else {
        auto deviceOri = at::Device(std::string(device));
        std::string device_str = "meta";
        if (deviceOri.has_index()) {
            device_str += ":";
            device_str += std::to_string(deviceOri.index());
        }
        output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(at::Device(device_str)));
    }
    return output;
}
```

并在 `torch_binding_meta.cpp` 的 `init` block 注册:

```cpp
    ops.impl("npu_fused_infer_attention_sink_metadata",
             &vllm_ascend::meta::npu_fused_infer_attention_sink_metadata_meta);
```

### 5.3 命名空间变化

| 项 | omni-ops | vllm-ascend |
|---|---|---|
| Torch op 命名空间 | `torch.ops.custom._npu_fused_infer_attention_sink_metadata` | `torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata` |
| 名称前导下划线 | 有(`_npu_...`) | 无(`npu_...`,与 `npu_sparse_attn_sharedkv_metadata` 一致) |

> 调用方(omni-ops 文档示例 `torch.ops.custom._npu_fused_infer_attention_sink_metadata(...)`)需同步改为 `torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(...)`。

## 6. 构建系统接入

### 6.1 `csrc/build_aclnn.sh`

在 `CUSTOM_OPS_ARRAY` 中添加新算子。该算子支持 A2/A3,需加入 `ascend910b` 与 `ascend910_93` 两个分支(若后续支持 A5 再加入 `ascend950`):

```bash
# ascend910b 分支
CUSTOM_OPS_ARRAY=(
    ...
    "sparse_attn_sharedkv_metadata"
    "ai_infra_fused_infer_attention_sink_metadata"   # 新增
    ...
)

# ascend910_93 分支
CUSTOM_OPS_ARRAY=(
    ...
    "sparse_attn_sharedkv_metadata"
    "ai_infra_fused_infer_attention_sink_metadata"   # 新增
    ...
)
```

### 6.2 安装产物

构建后安装到 `vllm_ascend/_cann_ops_custom`,由 `build_aclnn.sh` 现有逻辑统一处理,无需额外改动。

## 7. Python 侧接入

### 7.1 `vllm_ascend/device/device_op.py`

该算子作为 FIA-Sink 的前置 metadata,需在 `device_op.py` 暴露 op selector。参考 `get_dsa_sparse_attn_metadata_op()` 模式,在对应 device op 类中新增:

```python
@staticmethod
def get_fia_sink_metadata_op():
    """Returns the metadata-building operator for fused infer attention sink."""
    return torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata

@staticmethod
def get_fia_sink_metadata_kwargs(device):
    """Returns kwargs for fia sink metadata builder."""
    return {"device": str(device)}
```

> 具体落在哪个 device op 类,取决于 FIA-Sink 主算子(`npu_fused_infer_attention_sink`)的接入位置。若主算子尚未迁移到 vllm-ascend,本 metadata 算子的 Python selector 可暂缓,仅完成 C++ 侧注册与构建,待主算子迁移时一并接入。

## 8. 迁移步骤清单

1. **新建目录** `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/` 及子目录。
2. **迁移 CANN 算子文件**(第 4 节映射表),调整 `#include` 路径:
   - `aicpu.cpp` 中 `#include "../../ai_infra_fused_infer_attention_sink/op_kernel/ai_infra_fused_infer_attention_sink_metadata.h"` → `#include "ai_infra_fused_infer_attention_sink_metadata.h"`(本目录)
   - `infershape.cpp` 中同上
   - 将该共享头文件复制到 `op_kernel_aicpu/` 下
3. **新增 CMakeLists.txt**(顶层 + `op_kernel_aicpu/`)。
4. **新增 torch 绑定**:`torch_binding.cpp`(NPU impl + def/impl)、`torch_binding_meta.cpp`(Meta impl + impl)。
5. **修改 `build_aclnn.sh`**:加入 `CUSTOM_OPS_ARRAY`。
6. **新增 README.md**(可选,参考 `sparse_attn_sharedkv_metadata/README.md`)。
7. **构建验证**:`bash build_aclnn.sh <ROOT> <SOC_VERSION>`,确认 `vllm_ascend/_cann_ops_custom` 下生成算子库。
8. **Python 验证**:`torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(...)` 可调用并返回 `[1024]` int32 张量。
9. **精度/分核验证**:与 omni-ops 原算子在相同输入下输出一致(对比 metadata 张量)。

## 9. 关键改动代码汇总(变更点)

| 文件 | 操作 | 说明 |
|---|---|---|
| `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/**` | 新增 | 整个算子目录(第 4 节) |
| `csrc/torch_binding.cpp` | 修改 | 新增 `npu_fused_infer_attention_sink_metadata_npu` + `ops.def/ops.impl` |
| `csrc/torch_binding_meta.cpp` | 修改 | 新增 `_meta` 函数 + `ops.impl` |
| `csrc/build_aclnn.sh` | 修改 | `CUSTOM_OPS_ARRAY` 加入新算子(a2/a3 分支) |
| `vllm_ascend/device/device_op.py` | 修改(可选) | 新增 op selector(待主算子接入时) |

## 10. 风险与注意事项

1. **共享头文件归属**:源码中 `ai_infra_fused_infer_attention_sink_metadata.h` 原属主 FIA-Sink 算子目录,被 metadata 算子的 infershape/aicpu 引用。迁移后将其放入 metadata 算子目录,避免跨算子引用;若后续主 FIA-Sink 算子也迁移到 vllm-ascend,需协调该头文件的归属(建议保留在 metadata 算子侧,主算子反向引用,因为 metadata 结构定义归属 metadata 算子)。
2. **AICORE 核数默认值**:omni-ops torch 绑定默认 `aic_core_num=24, aiv_core_num=48`;aclnn `GetWorkspaceSize` 内部会根据平台信息校正(<=0 或超过实际核数时取实际值)。vllm-ascend 侧保持相同默认值与校正逻辑,无需改动。
3. **V1/V2 分支**:`batch_invariant=true` 走 V2,`false` 走 V1。两个 ACLNN 共用同一 l0 算子与 AICPU kernel,迁移时务必保持分支选择与 omni-ops 一致(见 5.2.1)。
4. **不迁移 torchair GE converter**:vllm-ascend 的图模式走 aclgraph,而非 torchair `register_fx_node_ge_converter`。omni-ops `converter/npu_fused_infer_attention_sink_metadata.py` 不迁移。若后续 vllm-ascend 有 torchair 图模式需求再评估。
5. **算子命名**:遵循 vllm-ascend 惯例去除前导下划线(`npu_fused_infer_attention_sink_metadata`),调用方需同步更新。
6. **AGENTS.md 合规**:
   - 命名:函数 `snake_case`、类 `PascalCase`、常量 `ALL_UPPER_CASE`(源码已符合)。
   - 避免在热路径 `tensor.item()`(本算子为 metadata 计算,无此问题)。
   - 环境变量:本算子不引入新环境变量。
   - 测试:需补充 `tests/ut/` 与 `tests/e2e/`(见第 11 节)。
   - No Magic Numbers:`OUTPUT_SIZE = 1024` 已用具名常量表达(值与源码 `FIASINK_META_SIZE` 对齐;参考实现 `npu_sparse_attn_sharedkv_metadata_npu` 同样使用 `OUTPUT_SIZE = 1024`,保持惯例一致)。算子签名默认值 `pre_tokens=2147483647`/`next_tokens=2147483647` 为接口契约的一部分(语义为 INT32_MAX,表示无前/后向 token 限制),需与 omni-ops 保持一致以保证行为兼容,不另设常量。
   - V1/V2 参数顺序:已与 `aclnnAiInfraFusedInferAttentionSinkMetadataGetWorkspaceSize`(V1,18 参数)与 `...V2GetWorkspaceSize`(V2,19 参数,多 `batchInvariant`)头文件签名逐一核对,`EXEC_NPU_CMD` 调用顺序正确(V1/V2 均不传 `socVersion`,由 ACLNN 内部 `GetCurrentPlatformInfo` 获取)。

## 11. 测试计划

1. **UT(算子层)**:在 `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/tests/ut/` 下迁移 omni-ops 的 `op_api`/`op_host` 单测(若 vllm-ascend 的 aicpu CMake 启用 `UT_TEST_ALL`/`OP_KERNEL_AICPU_UT`,使用 `AddAicpuOpTestCase` 注册)。
2. **ST(端到端)**:在 `vllm_ascend/tests/e2e/` 下新增 `test_npu_fused_infer_attention_sink_metadata.py`,覆盖:
   - 默认参数(GQA, TND, sparse_mode=0)
   - `batch_invariant=True`(走 V2)
   - sink_num > 0 / k_sink_num > 0
   - 与 omni-ops 输出一致性对比
3. **回归**:确认 `build_aclnn.sh` 在 `ascend910b` 与 `ascend910_93` 均构建成功。
