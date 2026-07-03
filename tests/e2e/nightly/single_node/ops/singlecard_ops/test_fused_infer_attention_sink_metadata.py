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
"""E2E system tests for npu_fused_infer_attention_sink_metadata operator.

This operator computes metadata (core-split info) for the FIA-Sink attention
kernel. It has two ACLNN backends:
  - V1 (batch_invariant=False): aclnnAiInfraFusedInferAttentionSinkMetadata
  - V2 (batch_invariant=True):  aclnnAiInfraFusedInferAttentionSinkMetadataV2

Test coverage:
  - Output shape and dtype (must be [1024] int32)
  - V1 path (batch_invariant=False)
  - V2 path (batch_invariant=True)
  - V1/V2 consistency when sink_num=0
  - Various sink_num / layout / batch_size combinations
  - Metadata is not all zeros (kernel actually executed)

Prerequisites:
  - NPU environment (Ascend 910B or 910C)
  - Custom ops built and installed via: bash csrc/build_aclnn.sh . <soc_version>
"""

import unittest

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Output size constant (must match C++ definition)
EXPECTED_OUTPUT_SIZE = 1024


class TestFusedInferAttentionSinkMetadata(unittest.TestCase):
    """ST for npu_fused_infer_attention_sink_metadata."""

    def setUp(self):
        torch.manual_seed(42)
        self.device_id = 0
        torch.npu.set_device(self.device_id)
        self.npu = f"npu:{self.device_id}"

    def _call_op(self, **overrides):
        """Helper to call the op with sensible defaults and overrides."""
        defaults = dict(
            num_heads_q=8,
            num_heads_kv=8,
            head_dim_qk=128,
            head_dim_v=128,
            actual_seq_lengths=torch.tensor([128, 64], dtype=torch.int64, device=self.npu),
            actual_seq_lengths_kv=torch.tensor([128, 64], dtype=torch.int64, device=self.npu),
            batch_size=2,
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            input_layout="TND",
            input_layout_kv="TND",
            sink_num=0,
            k_sink_num=0,
            batch_invariant=False,
            rope_head_dim=0,
            block_size=0,
            aic_core_num=24,
            aiv_core_num=48,
        )
        defaults.update(overrides)
        return torch.ops._C_ascend.npu_fused_infer_attention_sink_metadata(**defaults)

    # ------------------------------------------------------------------
    # Output shape / dtype verification
    # ------------------------------------------------------------------

    def test_output_shape_v1(self):
        """V1: output must be [1024] int32."""
        metadata = self._call_op(batch_invariant=False)
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))
        self.assertEqual(metadata.dtype, torch.int32)

    def test_output_shape_v2(self):
        """V2: output must be [1024] int32."""
        metadata = self._call_op(batch_invariant=True)
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))
        self.assertEqual(metadata.dtype, torch.int32)

    def test_output_not_all_zeros_v1(self):
        """V1: metadata should not be all zeros (kernel executed)."""
        metadata = self._call_op(batch_invariant=False)
        self.assertNotEqual(metadata.sum().item(), 0,
                            "metadata is all zeros, kernel may not have executed")

    def test_output_not_all_zeros_v2(self):
        """V2: metadata should not be all zeros (kernel executed)."""
        metadata = self._call_op(batch_invariant=True)
        self.assertNotEqual(metadata.sum().item(), 0,
                            "metadata is all zeros, kernel may not have executed")

    # ------------------------------------------------------------------
    # V1/V2 consistency
    # ------------------------------------------------------------------

    def test_v1_v2_consistency_sink_zero(self):
        """When sink_num=0, V1 and V2 should produce identical output."""
        v1 = self._call_op(batch_invariant=False, sink_num=0, k_sink_num=0)
        v2 = self._call_op(batch_invariant=True, sink_num=0, k_sink_num=0)
        self.assertTrue(torch.equal(v1.cpu(), v2.cpu()),
                        "V1 and V2 output mismatch when sink_num=0")

    # ------------------------------------------------------------------
    # Parametrized scenarios
    # ------------------------------------------------------------------

    def test_v1_single_batch(self):
        """V1 with batch_size=1."""
        metadata = self._call_op(
            batch_invariant=False,
            batch_size=1,
            actual_seq_lengths=torch.tensor([128], dtype=torch.int64, device=self.npu),
            actual_seq_lengths_kv=torch.tensor([128], dtype=torch.int64, device=self.npu),
        )
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))

    def test_v2_with_sink(self):
        """V2 with sink_num=128, k_sink_num=128, rope_head_dim=64."""
        metadata = self._call_op(
            batch_invariant=True,
            sink_num=128,
            k_sink_num=128,
            rope_head_dim=64,
            block_size=128,
        )
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))
        self.assertEqual(metadata.dtype, torch.int32)

    def test_v2_bsh_layout(self):
        """V2 with BSH layout."""
        metadata = self._call_op(
            batch_invariant=True,
            input_layout="BSH",
            input_layout_kv="BSH",
        )
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))

    def test_v1_no_optional_tensors(self):
        """V1 without actual_seq_lengths (pass None)."""
        metadata = self._call_op(
            batch_invariant=False,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=None,
            batch_size=2,
        )
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))

    def test_v2_large_batch(self):
        """V2 with batch_size=4."""
        metadata = self._call_op(
            batch_invariant=True,
            batch_size=4,
            actual_seq_lengths=torch.tensor([128, 64, 32, 16], dtype=torch.int64, device=self.npu),
            actual_seq_lengths_kv=torch.tensor([128, 64, 32, 16], dtype=torch.int64, device=self.npu),
        )
        self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))

    def test_different_core_nums(self):
        """V1 with different aic/aiv core num combinations."""
        for aic, aiv in [(24, 48), (36, 72)]:
            metadata = self._call_op(
                batch_invariant=False,
                aic_core_num=aic,
                aiv_core_num=aiv,
            )
            self.assertEqual(metadata.shape, torch.Size([EXPECTED_OUTPUT_SIZE]))


if __name__ == "__main__":
    unittest.main()
