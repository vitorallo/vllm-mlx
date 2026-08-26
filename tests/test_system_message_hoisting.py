# SPDX-License-Identifier: Apache-2.0
"""Non-leading system messages must be hoisted, not passed through.

Qwen 3.5 / 3.8 chat templates raise
``TemplateError("System message must be at the beginning.")`` on a system
message at any index but 0, which reaches the client as an opaque HTTP 500.
Claude Code sends such messages (and "developer" maps to "system", producing
more), so the whole session dies on a template exception.
"""

import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="server import requires Apple Silicon deps",
)


def _norm(messages):
    from vllm_mlx.server import _normalize_messages

    return _normalize_messages(messages)


def test_late_system_is_merged_into_leading_system():
    out = _norm(
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be terse"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "base\n\nbe terse"
    assert out[1]["content"] == "hi"


def test_late_system_with_no_leading_system_becomes_leading():
    out = _norm(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be terse"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "be terse"


def test_multiple_late_systems_preserved_in_order():
    out = _norm(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "b"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "c"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[0]["content"] == "a\n\nb\n\nc"


def test_developer_role_late_is_also_hoisted():
    """developer -> system happens first, so it hits the same path."""
    out = _norm(
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "dev note"},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "base\n\ndev note"


def test_ordinary_conversation_is_untouched():
    msgs = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    out = _norm([m.copy() for m in msgs])
    assert out == msgs


def test_multimodal_late_system_is_demoted_not_dropped():
    blocks = [{"type": "text", "text": "img note"}]
    out = _norm(
        [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": blocks},
        ]
    )
    assert [m["role"] for m in out] == ["system", "user", "user"]
    assert out[-1]["content"] == blocks
