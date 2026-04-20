# Copyright 2026 The RLinf Authors.
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

"""Tests for env worker initialization helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_INIT_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "workers"
    / "env"
    / "init_utils.py"
)
_INIT_UTILS_SPEC = importlib.util.spec_from_file_location(
    "rlinf_env_init_utils", _INIT_UTILS_PATH
)
assert _INIT_UTILS_SPEC is not None
assert _INIT_UTILS_SPEC.loader is not None
init_utils = importlib.util.module_from_spec(_INIT_UTILS_SPEC)
_INIT_UTILS_SPEC.loader.exec_module(init_utils)

resolve_env_classes = init_utils.resolve_env_classes


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(env_type="wan_wm"),
            eval=SimpleNamespace(env_type="libero"),
        )
    )


def test_resolve_env_classes_skips_eval_import_when_eval_disabled():
    cfg = _make_cfg()
    calls = []

    def fake_get_env_cls(env_type, env_cfg):
        calls.append((env_type, env_cfg))
        return "train-env"

    train_env_cls, eval_env_cls = resolve_env_classes(
        cfg, enable_eval=False, get_env_cls_fn=fake_get_env_cls
    )

    assert train_env_cls == "train-env"
    assert eval_env_cls is None
    assert calls == [("wan_wm", cfg.env.train)]


def test_resolve_env_classes_imports_eval_when_enabled():
    cfg = _make_cfg()
    calls = []

    def fake_get_env_cls(env_type, env_cfg):
        calls.append((env_type, env_cfg))
        return "train-env" if env_type == "wan_wm" else "eval-env"

    train_env_cls, eval_env_cls = resolve_env_classes(
        cfg, enable_eval=True, get_env_cls_fn=fake_get_env_cls
    )

    assert train_env_cls == "train-env"
    assert eval_env_cls == "eval-env"
    assert calls == [
        ("wan_wm", cfg.env.train),
        ("libero", cfg.env.eval),
    ]
