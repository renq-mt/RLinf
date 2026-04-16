from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "openvla_oft"
    / "attn_utils.py"
)
MODULE_SPEC = spec_from_file_location("test_attn_utils_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
attn_utils = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(attn_utils)


def test_resolve_attn_implementation_keeps_default_backend_off_musa(monkeypatch):
    monkeypatch.setattr(attn_utils, "is_musa_available", lambda: False)

    assert (
        attn_utils.resolve_attn_implementation(
            {"attn_implementation": "flash_attention_2"}
        )
        is None
    )


def test_resolve_attn_implementation_downgrades_flash_attn_to_sdpa_on_musa(
    monkeypatch,
):
    monkeypatch.setattr(attn_utils, "is_musa_available", lambda: True)

    assert (
        attn_utils.resolve_attn_implementation(
            {"attn_implementation": "flash_attention_2"}
        )
        == "sdpa"
    )


def test_resolve_attn_implementation_uses_eager_only_when_explicitly_requested(
    monkeypatch,
):
    monkeypatch.setattr(attn_utils, "is_musa_available", lambda: True)

    assert (
        attn_utils.resolve_attn_implementation(
            {
                "attn_implementation": "flash_attention_2",
                "force_musa_eager_attn_implementation": True,
            }
        )
        == "eager"
    )
