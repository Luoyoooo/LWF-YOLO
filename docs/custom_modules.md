# Custom Module Map

This document maps the paper modules to their code implementations.

## Paper Module Mapping

| Paper name | Code classes | File | Line range |
| --- | --- | --- | --- |
| MSEFE | SobelConv + ScaleEdge + EdgeFusion + MSEFE | lwf_modules.py | See section comments |
| DCGFormer | LayerNormGeneral + DynamicConvGLU + DCGFormerBlock + DCGFormerC3k + DCGFormer | lwf_modules.py | See section comments |
| DyDCN | DyDCNBlock + Detect_DyDCN + DyDCN(alias) | lwf_modules.py | See section comments |

MSEFE corresponds to the combination of ScaleEdge for edge extraction and EdgeFusion for feature fusion, with MSEFE exposed as the unified entry point. DyDCN corresponds to DyDCNBlock as the core computation module and Detect_DyDCN as the detection-head wrapper.
