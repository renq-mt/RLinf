# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Any

import torch


def is_musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def resolve_attn_implementation(cfg: Mapping[str, Any]) -> str | None:
    """Resolve attention implementation override for the current device/runtime."""
    force_musa_eager_attn_implementation = bool(
        cfg.get("force_musa_eager_attn_implementation", False)
    )

    if is_musa_available() and force_musa_eager_attn_implementation:
        return "eager"

    if is_musa_available() and cfg.get("attn_implementation") == "flash_attention_2":
        return "sdpa"

    # Match upstream CUDA behavior by not forcing an attention backend unless
    # the MUSA fallback is explicitly needed.
    return None
