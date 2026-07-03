/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ai_infra_fused_infer_attention_sink_metadata_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "../op_kernel_aicpu/ai_infra_fused_infer_attention_sink_metadata.h"

using namespace ge;

namespace {
static ge::graphStatus InferShapeAiInfraFusedInferAttentionSinkMetadata(gert::InferShapeContext* context)
{
    gert::Shape* oShape = context->GetOutputShape(0);
    // output shape (FIASINK_METADATA_T, )
    oShape->SetDimNum(1);
    oShape->SetDim(0, optiling::FIASINK_META_SIZE);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeAiInfraFusedInferAttentionSinkMetadata(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AiInfraFusedInferAttentionSinkMetadata)
    .InferShape(InferShapeAiInfraFusedInferAttentionSinkMetadata)
    .InferDataType(InferDtypeAiInfraFusedInferAttentionSinkMetadata);
} // namespace
