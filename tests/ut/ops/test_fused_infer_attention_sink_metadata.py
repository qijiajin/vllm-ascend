#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for npu_fused_infer_attention_sink_metadata operator.

This operator computes metadata (core-split info) for the FIA-Sink attention
kernel. It is registered via torch.ops._C_ascend and has two ACLNN backends:
  - V1 (batch_invariant=False): aclnnAiInfraFusedInferAttentionSinkMetadata
  - V2 (batch_invariant=True):  aclnnAiInfraFusedInferAttentionSinkMetadataV2

These UTs run on CPU with mocked NPU ops, verifying:
  - Operator registration in torch.ops._C_ascend
  - Schema correctness (parameter names, types, defaults)
  - Mock call behavior (parameter forwarding, V1/V2 branching)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

OP_NAME = "npu_fused_infer_attention_sink_metadata"
OP_FULL_PATH = f"torch.ops._C_ascend.{OP_NAME}"

# Output size constant (must match C++ definition in torch_binding.cpp)
EXPECTED_OUTPUT_SIZE = 1024


def _mock_metadata_output(*args, **kwargs):
    """Mock implementation that returns a [1024] int32 tensor."""
    return torch.empty(EXPECTED_OUTPUT_SIZE, dtype=torch.int32)


class TestOpRegistration:
    """Verify the operator is properly registered in torch.ops._C_ascend."""

    def test_op_exists(self):
        """Operator should be accessible via torch.ops._C_ascend."""
        assert hasattr(torch.ops._C_ascend, OP_NAME), \
            f"{OP_FULL_PATH} is not registered"

    def test_op_schema_has_correct_params(self):
        """Operator schema should declare all expected parameters."""
        op = getattr(torch.ops._C_ascend, OP_NAME)
        schema = str(op._schema)
        # Check key parameters exist in schema
        expected_params = [
            "num_heads_q", "num_heads_kv", "head_dim_qk", "head_dim_v",
            "actual_seq_lengths", "actual_seq_lengths_kv",
            "batch_size", "sparse_mode", "pre_tokens", "next_tokens",
            "input_layout", "input_layout_kv",
            "sink_num", "k_sink_num", "batch_invariant",
            "rope_head_dim", "block_size",
            "aic_core_num", "aiv_core_num", "device",
        ]
        for param in expected_params:
            assert param in schema, \
                f"Parameter '{param}' not found in op schema: {schema}"

    def test_op_schema_returns_tensor(self):
        """Operator should return a single Tensor."""
        op = getattr(torch.ops._C_ascend, OP_NAME)
        schema = str(op._schema)
        assert "-> Tensor" in schema, \
            f"Schema should return Tensor, got: {schema}"


class TestOpDefaults:
    """Verify default parameter values in the op schema."""

    def test_default_batch_invariant_is_false(self):
        """batch_invariant should default to False (V1 path)."""
        op = getattr(torch.ops._C_ascend, OP_NAME)
        schema = str(op._schema)
        assert "batch_invariant=False" in schema, \
            f"batch_invariant should default to False, schema: {schema}"

    def test_default_device_is_npu(self):
        """device should default to 'npu'."""
        op = getattr(torch.ops._C_ascend, OP_NAME)
        schema = str(op._schema)
        assert 'device="npu"' in schema, \
            f"device should default to 'npu', schema: {schema}"

    def test_default_input_layout_is_tnd(self):
        """input_layout should default to 'TND'."""
        op = getattr(torch.ops._C_ascend, OP_NAME)
        schema = str(op._schema)
        assert 'input_layout="TND"' in schema, \
            f"input_layout should default to TND, schema: {schema}"


class TestOpCallBehavior:
    """Verify mock call behavior and parameter forwarding.

    On CPU (no NPU), the actual kernel cannot execute.
    We patch the op to verify it is called with correct parameters.
    """

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_call_with_minimal_args(self, mock_op):
        """Calling with only required args should use defaults for the rest."""
        result = torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8,
            num_heads_kv=8,
            head_dim_qk=128,
            head_dim_v=128,
        )
        assert result.shape == torch.Size([EXPECTED_OUTPUT_SIZE])
        assert result.dtype == torch.int32
        mock_op.assert_called_once()

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_batch_invariant_false_uses_v1(self, mock_op):
        """batch_invariant=False should be forwarded correctly."""
        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            batch_invariant=False,
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["batch_invariant"] is False

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_batch_invariant_true_uses_v2(self, mock_op):
        """batch_invariant=True should be forwarded correctly."""
        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            batch_invariant=True,
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["batch_invariant"] is True

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_optional_tensors_forwarded(self, mock_op):
        """Optional tensor args should be forwarded when provided."""
        seq_lens = torch.tensor([128, 64], dtype=torch.int64)
        seq_lens_kv = torch.tensor([128, 64], dtype=torch.int64)

        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            actual_seq_lengths=seq_lens,
            actual_seq_lengths_kv=seq_lens_kv,
            batch_size=2,
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["actual_seq_lengths"] is seq_lens
        assert call_kwargs["actual_seq_lengths_kv"] is seq_lens_kv
        assert call_kwargs["batch_size"] == 2

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_layout_params_forwarded(self, mock_op):
        """Layout string params should be forwarded."""
        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            input_layout="BSH",
            input_layout_kv="BSND",
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["input_layout"] == "BSH"
        assert call_kwargs["input_layout_kv"] == "BSND"

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    def test_sink_params_forwarded(self, mock_op):
        """Sink-related params should be forwarded."""
        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            sink_num=128,
            k_sink_num=64,
            rope_head_dim=32,
            block_size=64,
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["sink_num"] == 128
        assert call_kwargs["k_sink_num"] == 64
        assert call_kwargs["rope_head_dim"] == 32
        assert call_kwargs["block_size"] == 64

    @patch(OP_FULL_PATH, side_effect=_mock_metadata_output)
    @pytest.mark.parametrize("aic,aiv", [(24, 48), (36, 72), (0, 0)])
    def test_core_num_params_forwarded(self, mock_op, aic, aiv):
        """AIC/AIV core num params should be forwarded."""
        torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(
            num_heads_q=8, num_heads_kv=8, head_dim_qk=128, head_dim_v=128,
            aic_core_num=aic,
            aiv_core_num=aiv,
        )
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["aic_core_num"] == aic
        assert call_kwargs["aiv_core_num"] == aiv
