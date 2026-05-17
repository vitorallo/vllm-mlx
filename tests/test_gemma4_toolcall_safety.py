# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool-call safety in Gemma 4 channel cleaning + the opt-in
truncated-tool-call notice.

Regression coverage for: a `Write`/`Edit` tool call (whose JSON contains a
whole file body) being silently destroyed because
  D1) an unclosed `<|channel>thought` slice deleted to end-of-text, taking a
      following tool call with it, and
  D2) a `<|channel>thought` / `<channel|>` substring *inside* the tool-call
      JSON content triggered that destructive slice / regex sub.
Plus the model-agnostic, opt-in "write it in parts" notice that replaces the
prior silent HTTP-200-text on max_tokens truncation.

No MLX dependency for the channel tests; server import guarded to arm64 to
match repo convention.
"""

import json
import platform
import sys

import pytest

from vllm_mlx.api.utils import _clean_gemma4_channels
from vllm_mlx.tool_parsers.auto_tool_parser import AutoToolParser

_FILE_BODY = "\n".join(f"def f{i}(): return {i}" for i in range(60))
_WRITE_ARGS = json.dumps({"file_path": "/tmp/x.py", "content": _FILE_BODY})


def _parses_to_tool_call(raw: str) -> bool:
    """Run text through the channel cleaner then the auto parser."""
    return AutoToolParser().extract_tool_calls(_clean_gemma4_channels(raw)).tools_called


class TestGemma4ChannelToolCallSafety:
    """The channel cleaner must never corrupt a tool-call span."""

    def test_complete_thought_then_complete_call_parses(self):
        raw = f"<|channel>thought\nplan<channel|><|tool_call>call:Write{_WRITE_ARGS}<tool_call|>"
        assert _parses_to_tool_call(raw) is True

    def test_d1_unclosed_thought_then_complete_call_preserved(self):
        # Truncated/unclosed thought immediately followed by a full tool call:
        # the tool call must survive (was deleted-to-EOF before the fix).
        raw = f"<|channel>thought reasoning got cut <|tool_call>call:Write{_WRITE_ARGS}<tool_call|>"
        assert _parses_to_tool_call(raw) is True

    def test_d2_channel_marker_inside_gemma_content_preserved(self):
        poisoned = json.dumps(
            {
                "file_path": "/tmp/doc.md",
                "content": "# Notes on <|channel>thought tokens\nand <channel|>\n"
                + _FILE_BODY,
            }
        )
        raw = f"<|tool_call>call:Write{poisoned}<tool_call|>"
        assert _parses_to_tool_call(raw) is True

    def test_d2_channel_marker_inside_qwen_content_preserved(self):
        poisoned = json.dumps(
            {"file_path": "/tmp/doc.md", "content": "see <|channel>thought x"}
        )
        raw = f'<tool_call>{{"name": "Write", "arguments": {poisoned}}}</tool_call>'
        assert _parses_to_tool_call(raw) is True

    def test_multiple_tool_call_spans_preserved_verbatim(self):
        # The cleaner's contract is span preservation (not multi-extraction,
        # which is the parser's concern). Both tool-call spans must survive
        # cleaning byte-for-byte, and the thought is still stripped.
        a = json.dumps({"file_path": "/a", "content": "aaa"})
        b = json.dumps({"file_path": "/b", "content": "bbb"})
        span_a = f"<|tool_call>call:Write{a}<tool_call|>"
        span_b = f"<|tool_call>call:Write{b}<tool_call|>"
        cleaned = _clean_gemma4_channels(
            f"<|channel>thought plan<channel|>{span_a}{span_b}"
        )
        assert span_a in cleaned
        assert span_b in cleaned
        assert "thought plan" not in cleaned

    def test_truncated_mid_json_still_unparseable(self):
        # Genuinely incomplete JSON: cleaner must not magically "fix" it; the
        # opt-in notice (separate path) is what surfaces this case.
        raw = '<|channel>thought w<|tool_call>call:Write{"file_path":"/x","content":"def f0():'
        assert _parses_to_tool_call(raw) is False

    # --- Regression: behaviour with NO tool call must be byte-identical ---

    def test_plain_truncated_thought_stripped(self):
        assert _clean_gemma4_channels("<|channel>thought never closes") == ""

    def test_complete_thought_plus_text_unwrapped(self):
        assert (
            _clean_gemma4_channels("<|channel>thought hidden<channel|>Hello world")
            == "Hello world"
        )

    def test_no_channel_no_tool_passthrough(self):
        assert _clean_gemma4_channels("just a normal answer") == "just a normal answer"


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="server import requires Apple Silicon deps",
)
class TestTruncatedToolCallNotice:
    """Opt-in, model-agnostic, default-off. Must not affect other consumers."""

    def _fn(self):
        from vllm_mlx.server import _truncated_tool_call_notice

        return _truncated_tool_call_notice

    def test_default_off_returns_none(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = False
        assert self._fn()("<|tool_call>call:Write{", "length", False) is None

    def test_on_length_marker_no_call_returns_message(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            msg = self._fn()("x <|tool_call>call:Write{", "length", False)
            assert msg is not None
            assert "incrementally" in msg and "smaller" in msg
        finally:
            s._tool_call_truncation_notice = False

    def test_on_but_had_tool_calls_returns_none(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()("<|tool_call>x", "length", True) is None
        finally:
            s._tool_call_truncation_notice = False

    def test_on_but_not_length_returns_none(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()("<|tool_call>x", "stop", False) is None
        finally:
            s._tool_call_truncation_notice = False

    def test_on_length_no_marker_returns_none(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()("plain text only", "length", False) is None
        finally:
            s._tool_call_truncation_notice = False

    def test_model_agnostic_qwen_marker(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()('<tool_call>{"name"', "length", False) is not None
        finally:
            s._tool_call_truncation_notice = False
