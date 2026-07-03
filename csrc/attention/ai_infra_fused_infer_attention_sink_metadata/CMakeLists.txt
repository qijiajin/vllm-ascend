# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
# Migrated from omni-ops. The shared metadata header (ai_infra_fused_infer_attention_sink_metadata.h)
# is colocated in op_kernel_aicpu/ to avoid cross-operator directory dependencies, so no DEPENDENCIES
# on the (not-yet-migrated) FIA-Sink main operator is required.
add_modules_sources_aicpu()
