# _npu_fused_infer_attention_sink_metadata 算子迁移操作记录

> 迁移日期: 2026-07-03
> 源仓库: `omni-ops` (`inference/ascendc/src/ops-transformer/attention/ai_infra_fused_infer_attention_sink_metadata/`)
> 目标仓库: `vllm-ascend` (`csrc/attention/ai_infra_fused_infer_attention_sink_metadata/`)
> 参考实现: `csrc/attention/sparse_attn_sharedkv_metadata/`
> 方案文档: `csrc/attention/migration_plan_fused_infer_attention_sink_metadata.md`

---

## 1. 新增文件(13 个)

### 1.1 算子目录 `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/`

| 子目录 | 文件 | 来源 |
|---|---|---|
| `op_api/` | `aclnn_ai_infra_fused_infer_attention_sink_metadata.h` | omni-ops 原样复制 |
| `op_api/` | `aclnn_ai_infra_fused_infer_attention_sink_metadata.cpp` | omni-ops 原样复制 |
| `op_api/` | `aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.h` | omni-ops 原样复制 |
| `op_api/` | `aclnn_ai_infra_fused_infer_attention_sink_metadata_v2.cpp` | omni-ops 原样复制 |
| `op_api/` | `l0_ai_infra_fused_infer_attention_sink_metadata.h` | omni-ops 原样复制 |
| `op_api/` | `l0_ai_infra_fused_infer_attention_sink_metadata.cpp` | omni-ops 原样复制 |
| `op_graph/` | `ai_infra_fused_infer_attention_sink_metadata_proto.h` | omni-ops 原样复制 |
| `op_host/` | `ai_infra_fused_infer_attention_sink_metadata_infershape.cpp` | omni-ops 复制 + `#include` 路径调整 |
| `op_kernel_aicpu/` | `ai_infra_fused_infer_attention_sink_metadata_aicpu.h` | omni-ops 原样复制 |
| `op_kernel_aicpu/` | `ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp` | omni-ops 复制 + `#include` 路径调整 |
| `op_kernel_aicpu/` | `ai_infra_fused_infer_attention_sink_metadata_aicpu.json` | omni-ops 原样复制 |
| `op_kernel_aicpu/` | `ai_infra_fused_infer_attention_sink_metadata.h` | 从 `ai_infra_fused_infer_attention_sink/op_kernel/` 迁入(共享头) |
| — | `CMakeLists.txt` | 新建 |

### 1.2 `#include` 路径调整(2 处)

| 文件 | 原路径 | 新路径 | 原因 |
|---|---|---|---|
| `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp:13` | `#include "../../ai_infra_fused_infer_attention_sink/op_kernel/ai_infra_fused_infer_attention_sink_metadata.h"` | `#include "ai_infra_fused_infer_attention_sink_metadata.h"` | 共享头迁入本目录,改为本目录引用 |
| `op_host/ai_infra_fused_infer_attention_sink_metadata_infershape.cpp:16` | `#include "../../ai_infra_fused_infer_attention_sink/op_kernel/ai_infra_fused_infer_attention_sink_metadata.h"` | `#include "../op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata.h"` | 共享头迁入 `op_kernel_aicpu/`,从 `op_host/` 引用需 `../op_kernel_aicpu/` 前缀 |

> `op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata_aicpu.cpp:14` 的 `#include "../../common/aicpu/cpu_context_util.h"` 保持不变,与参考实现 `sparse_attn_sharedkv_metadata` 一致(由构建系统解析)。

---

## 2. 修改文件(3 个)

### 2.1 `csrc/torch_binding.cpp`

**(a) 新增 NPU 实现函数**(在 `npu_sparse_attn_sharedkv_metadata_npu` 之后,约第 1230 行):

```cpp
at::Tensor npu_fused_infer_attention_sink_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim_qk, int64_t head_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv,
    int64_t sink_num, int64_t k_sink_num, bool batch_invariant,
    int64_t rope_head_dim, int64_t block_size,
    int64_t aic_core_num, int64_t aiv_core_num,
    const c10::string_view device)
```

- 根据 `batch_invariant` 分支选择 V1/V2 ACLNN:
  - `batch_invariant=true` → `EXEC_NPU_CMD(aclnnAiInfraFusedInferAttentionSinkMetadataV2, ...)`
  - `batch_invariant=false` → `EXEC_NPU_CMD(aclnnAiInfraFusedInferAttentionSinkMetadata, ...)`
- 复用已有的 `get_valid_tensor` lambda 处理 optional tensor
- 输出 `[1024]` int32 张量

**(b) 新增 `ops.def` + `ops.impl` 注册**(在 `npu_sparse_attn_sharedkv_metadata` 注册之后,约第 2600 行):

```cpp
ops.def("npu_fused_infer_attention_sink_metadata(...) -> Tensor");
ops.impl("npu_fused_infer_attention_sink_metadata", torch::kPrivateUse1,
         &vllm_ascend::npu_fused_infer_attention_sink_metadata_npu);
```

### 2.2 `csrc/torch_binding_meta.cpp`

**(a) 新增 Meta 实现函数**(在 `npu_sparse_attn_sharedkv_metadata_meta` 之后,约第 1011 行):

```cpp
at::Tensor npu_fused_infer_attention_sink_metadata_meta(...)
```

- 输出 `[1024]` int32 张量,device 从 optional tensor 或 `device` 参数推导
- fallback 路径使用 `meta` device(与参考实现一致)

**(b) 新增 `ops.impl` 注册**(约第 1755 行):

```cpp
ops.impl("npu_fused_infer_attention_sink_metadata",
         &vllm_ascend::meta::npu_fused_infer_attention_sink_metadata_meta);
```

### 2.3 `csrc/build_aclnn.sh`

在 `ascend910b`(A2)和 `ascend910_93`(A3)两个分支的 `CUSTOM_OPS_ARRAY` 中,`"sparse_attn_sharedkv_metadata"` 之后添加:

```bash
"ai_infra_fused_infer_attention_sink_metadata"
```

- A2 分支:第 107 行
- A3 分支:第 172 行
- A5 分支(`ascend950`):暂未加入(与 `sparse_attn_sharedkv_metadata` 一致,后者也未在 A5 分支中)

---

## 3. CMakeLists.txt(新建)

采用 `sparse_attn_sharedkv_metadata` 的简单模式,通过 `add_modules_sources_aicpu()` 自动编译 op_api/op_host/op_kernel_aicpu 的所有源文件:

```cmake
add_modules_sources_aicpu()
```

- 不带 `DEPENDENCIES`(共享头已迁入本目录,不依赖未迁移的 FIA-Sink 主算子)
- `add_modules_sources_aicpu` 自动处理:
  - op_api/ 下的 l0 和 aclnn 源文件(第 376-392 行)
  - op_host/ 下的 infershape 源文件(第 394-398 行)
  - op_kernel_aicpu/ 下的 aicpu kernel 源文件(第 417-427 行)

---

## 4. 命名空间变化

| 项 | omni-ops | vllm-ascend |
|---|---|---|
| Torch op 命名空间 | `torch.ops.custom._npu_fused_infer_attention_sink_metadata` | `torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata` |
| 前导下划线 | 有(`_npu_...`) | 无(`npu_...`,与 `npu_sparse_attn_sharedkv_metadata` 一致) |

---

## 5. 未迁移项

| 项 | 原因 |
|---|---|
| `converter/npu_fused_infer_attention_sink_metadata.py`(torchair GE converter) | vllm-ascend 图模式走 aclgraph,不走 torchair |
| `docs/` 目录 | 文档不参与编译,且 vllm-ascend 的 sparse_attn_sharedkv_metadata 也无此目录 |
| `tests/` 目录 | omni-ops 的 UT/ST 组织方式与 vllm-ascend 不同,后续按 vllm-ascend 惯例补充 |
| `config.ini` | 非编译必需 |

---

## 6. 测试

### 6.1 测试文件清单

| 类型 | 文件路径 | 运行环境 | 用例数 |
|---|---|---|---|
| Python UT | `tests/ut/ops/test_fused_infer_attention_sink_metadata.py` | CPU(自动 mock) | 13 |
| Python ST | `tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_infer_attention_sink_metadata.py` | NPU | 12 |

### 6.2 UT 测试(CPU 可跑)

**文件**: `tests/ut/ops/test_fused_infer_attention_sink_metadata.py`

**测试内容**: 在 CPU 环境下 mock `torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata`,验证算子注册、schema、参数传递。

| 测试类 | 用例 | 说明 |
|---|---|---|
| `TestOpRegistration` | `test_op_exists` | 算子在 `torch.ops._C_ascend` 中注册 |
| | `test_op_schema_has_correct_params` | schema 包含全部 20 个参数 |
| | `test_op_schema_returns_tensor` | 返回类型为 Tensor |
| `TestOpDefaults` | `test_default_batch_invariant_is_false` | `batch_invariant` 默认 False |
| | `test_default_device_is_npu` | `device` 默认 "npu" |
| | `test_default_input_layout_is_tnd` | `input_layout` 默认 "TND" |
| `TestOpCallBehavior` | `test_call_with_minimal_args` | 仅传必需参数时使用默认值 |
| | `test_batch_invariant_false_uses_v1` | V1 路径参数传递 |
| | `test_batch_invariant_true_uses_v2` | V2 路径参数传递 |
| | `test_optional_tensors_forwarded` | optional tensor 参数传递 |
| | `test_layout_params_forwarded` | layout 字符串参数传递 |
| | `test_sink_params_forwarded` | sink 相关参数传递 |
| | `test_core_num_params_forwarded` | aic/aiv core num 参数传递(parametrize 3 组) |

**执行命令**:

```bash
cd d:/q00852295/work/code/pangu/vllm-ascend/pangu_branch/vllm-ascend

# 运行全部 UT
pytest -sv tests/ut/ops/test_fused_infer_attention_sink_metadata.py

# 运行单个测试类
pytest -sv tests/ut/ops/test_fused_infer_attention_sink_metadata.py::TestOpRegistration

# 运行单个用例
pytest -sv tests/ut/ops/test_fused_infer_attention_sink_metadata.py::TestOpCallBehavior::test_batch_invariant_false_uses_v1
```

### 6.3 ST 测试(需 NPU 环境)

**文件**: `tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_infer_attention_sink_metadata.py`

**测试内容**: 在 NPU 环境下调用真实算子,验证输出 shape/dtype、V1/V2 路径、一致性、多场景覆盖。

| 用例 | 说明 |
|---|---|
| `test_output_shape_v1` | V1 输出 `[1024]` int32 |
| `test_output_shape_v2` | V2 输出 `[1024]` int32 |
| `test_output_not_all_zeros_v1` | V1 metadata 非全零(kernel 执行) |
| `test_output_not_all_zeros_v2` | V2 metadata 非全零(kernel 执行) |
| `test_v1_v2_consistency_sink_zero` | sink_num=0 时 V1/V2 输出一致 |
| `test_v1_single_batch` | V1 batch_size=1 |
| `test_v2_with_sink` | V2 sink_num=128, rope_head_dim=64 |
| `test_v2_bsh_layout` | V2 BSH layout |
| `test_v1_no_optional_tensors` | V1 不传 optional tensor |
| `test_v2_large_batch` | V2 batch_size=4 |
| `test_different_core_nums` | V1 不同 aic/aiv 组合(parametrize 2 组) |

**前置条件**: 算子已编译安装

```bash
# 1. 先构建算子
cd d:/q00852295/work/code/pangu/vllm-ascend/pangu_branch/vllm-ascend
bash csrc/build_aclnn.sh . ascend910b    # 或 ascend910_93
```

**执行命令**:

```bash
cd d:/q00852295/work/code/pangu/vllm-ascend/pangu_branch/vllm-ascend

# 运行全部 ST
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_infer_attention_sink_metadata.py

# 运行单个用例
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_infer_attention_sink_metadata.py::TestFusedInferAttentionSinkMetadata::test_v1_v2_consistency_sink_zero
```

### 6.4 手动冒烟测试(快速验证)

构建完成后,可直接在 Python 交互式环境中快速验证:

```python
import torch
import torch_npu

op = torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata
print(f"op: {op}")

# 构造输入
seq = torch.tensor([128, 64], dtype=torch.int64, device="npu:0")

# V1 路径
metadata = op(
    num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
    actual_seq_lengths=seq, actual_seq_lengths_kv=seq,
    batch_size=2, sparse_mode=0, pre_tokens=65535, next_tokens=65535,
    input_layout="TND", input_layout_kv="TND",
    sink_num=0, k_sink_num=0, batch_invariant=False,
)
print(f"V1: shape={metadata.shape}, dtype={metadata.dtype}")
assert metadata.shape == torch.Size([1024])
assert metadata.dtype == torch.int32

# V2 路径
metadata_v2 = op(
    num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
    actual_seq_lengths=seq, actual_seq_lengths_kv=seq,
    batch_size=2, sparse_mode=0, pre_tokens=65535, next_tokens=65535,
    input_layout="TND", input_layout_kv="TND",
    sink_num=0, k_sink_num=0, batch_invariant=True,
)
print(f"V2: shape={metadata_v2.shape}, dtype={metadata_v2.dtype}")
```

### 6.5 与 omni-ops 一致性对比(可选)

在同一 NPU 环境上,对比 omni-ops 和 vllm-ascend 的输出:

```python
import torch
import torch_npu

seq = torch.tensor([128, 64], dtype=torch.int64, device="npu:0")
common = dict(
    num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
    actual_seq_lengths=seq, actual_seq_lengths_kv=seq,
    batch_size=2, sparse_mode=0, pre_tokens=65535, next_tokens=65535,
    input_layout="TND", input_layout_kv="TND",
    sink_num=0, k_sink_num=0, batch_invariant=False,
    rope_head_dim=0, block_size=0, aic_core_num=24, aiv_core_num=48,
)

# omni-ops
import omni_custom_ops
omni_out = torch.ops.custom._npu_fused_infer_attention_sink_metadata(**common)

# vllm-ascend
vllm_out = torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(**common)

assert torch.equal(omni_out.cpu(), vllm_out.cpu()), "Output mismatch!"
print("Consistency check PASSED")
```

### 6.6 C++ UT(后续补充)

omni-ops 已有完整的 C++ gtest 用例,后续可迁移到 `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/tests/ut/`:

| 源文件(omni-ops) | 测试内容 | 用例数 |
|---|---|---|
| `tests/ut/op_api/test_aclnn_ai_infra_fused_infer_attention_sink_metadata.cpp` | ACLNN V1/V2 GetWorkspaceSize | 3 |
| `tests/ut/op_host/test_ai_infra_fused_infer_attention_sink_metadata_infershape.cpp` | InferShape + InferDataType | 2 |

**执行命令**(迁移后):

```bash
cd d:/q00852295/work/code/pangu/vllm-ascend/pangu_branch/vllm-ascend/csrc

# 编译 UT
bash build.sh --test            # 全部 UT
bash build.sh --opapi_test      # 仅 op_api UT
bash build.sh --ophost_test     # 仅 op_host UT

# 运行
./build/ut/transformer_op_api_ut
./build/ut/transformer_infershape_ut
```

---

## 7. 后续待办

- [ ] 在 NPU 环境执行 `bash build_aclnn.sh <ROOT> <SOC_VERSION>` 验证构建(A2/A3)
- [ ] 运行 UT: `pytest -sv tests/ut/ops/test_fused_infer_attention_sink_metadata.py`
- [ ] 运行 ST: `pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_infer_attention_sink_metadata.py`
- [ ] 与 omni-ops 原算子做输出一致性对比(见 6.5)
- [ ] 迁移 C++ gtest 用例到 `csrc/attention/ai_infra_fused_infer_attention_sink_metadata/tests/ut/`(见 6.6)
- [ ] 待 FIA-Sink 主算子迁移到 vllm-ascend 后,在 `vllm_ascend/device/device_op.py` 中添加 op selector
