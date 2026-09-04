from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


runner_path = Path("vllm_ascend/worker/model_runner_v1.py")
text = runner_path.read_text()

text = replace_once(
    text,
    """                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)
""",
    """                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1, skip_gdn_state_update=True)
""",
    "idle DP dummy call",
)

text = replace_once(
    text,
    """        cascade_attn_prefix_lens: list[list[int]] | None = None,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_descriptor: BatchDescriptor | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
""",
    """        cascade_attn_prefix_lens: list[list[int]] | None = None,
        skip_gdn_state_update: bool = False,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_descriptor: BatchDescriptor | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
""",
    "attention metadata signature",
)

text = replace_once(
    text,
    """            builder = attn_group.get_metadata_builder(ubid or 0)
            device_metadata_provider = (
""",
    """            builder = attn_group.get_metadata_builder(ubid or 0)
            is_gdn_noop = skip_gdn_state_update and isinstance(
                builder,
                GDNAttentionMetadataBuilder,
            )
            if is_gdn_noop:
                # Dummy-run callers coalesce this; fall back to the unpadded
                # batch size instead of asserting in the execute path.
                gdn_num_reqs = (
                    num_reqs if num_reqs_padded is None else num_reqs_padded
                )
                # Idle DP dummy: keep captured GDN tensor ranks for graph
                # replay/collectives. num_actual_tokens=0 is the kernel no-op
                # (ops slice mixed_qkv[:0]); query_start_loc is a zero-filled
                # prefix sized to the graph, not a real token schedule.
                common_attn_metadata = replace(
                    common_attn_metadata,
                    query_start_loc=self.gdn_query_start_loc.gpu[
                        : gdn_num_reqs + 1
                    ],
                    query_start_loc_cpu=self.gdn_query_start_loc.cpu[
                        : gdn_num_reqs + 1
                    ],
                    num_actual_tokens=0,
                    max_query_len=0,
                    is_prefilling=(
                        torch.zeros_like(common_attn_metadata.is_prefilling)
                        if common_attn_metadata.is_prefilling is not None
                        else None
                    ),
                )
            device_metadata_provider = (
""",
    "GDN no-op metadata block",
)

text = replace_once(
    text,
    """            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
""",
    """            if (
                use_spec_decode
                and isinstance(builder, GDNAttentionMetadataBuilder)
                and not is_gdn_noop
            ):
""",
    "GDN spec-decode guard",
)

text = replace_once(
    text,
    """        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
    """        profile_seq_lens: int | None = None,
        profile_cpp: bool = False,
        skip_gdn_state_update: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
""",
    "dummy-run signature",
)

text = replace_once(
    text,
    """                if self._has_gdn:
                    self.gdn_query_start_loc.np[1 : num_reqs_padded + 1] = cum_num_tokens
                    self.gdn_query_start_loc.copy_to_gpu()
""",
    """                if self._has_gdn:
                    if skip_gdn_state_update:
                        self.gdn_query_start_loc.np.fill(0)
                    else:
                        self.gdn_query_start_loc.np[
                            1 : num_reqs_padded + 1
                        ] = cum_num_tokens
                    self.gdn_query_start_loc.copy_to_gpu()
""",
    "dummy GDN query_start_loc",
)

text = replace_once(
    text,
    """                    num_scheduled_tokens_np=num_scheduled_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                )
""",
    """                    num_scheduled_tokens_np=num_scheduled_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    skip_gdn_state_update=skip_gdn_state_update,
                )
""",
    "dummy attention metadata call",
)
runner_path.write_text(text)

runner310_path = Path("vllm_ascend/_310p/model_runner_310p.py")
text310 = runner310_path.read_text()
text310 = replace_once(
    text310,
    """        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ):
""",
    """        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
        skip_gdn_state_update: bool = False,
    ):
""",
    "310P dummy-run signature",
)
text310 = replace_once(
    text310,
    """                    num_active_loras=num_active_loras,
                    profile_seq_lens=profile_seq_lens,
                )
""",
    """                    num_active_loras=num_active_loras,
                    profile_seq_lens=profile_seq_lens,
                    skip_gdn_state_update=skip_gdn_state_update,
                )
""",
    "310P super dummy-run call",
)
runner310_path.write_text(text310)

test_path = Path("tests/ut/worker/test_model_runner_v1.py")
test_text = test_path.read_text()
test_text = replace_once(
    test_text,
    """        runner._dummy_run.assert_called_once_with(1)
""",
    """        runner._dummy_run.assert_called_once_with(
            1,
            skip_gdn_state_update=True,
        )
""",
    "idle dummy unit expectation",
)
test_path.write_text(test_text)
