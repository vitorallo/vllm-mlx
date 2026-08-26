# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in truncated-tool-call notice.

Regression coverage for a `Write`/`Edit` tool call being truncated by
max_tokens so it never closes, no `tool_use` can be parsed, and the caller
silently receives HTTP 200 with unusable partial text. With the flag on, the
server instead returns an explicit "write it in parts" instruction the agent
can act on.

Model-agnostic and default-off: every other consumer and every non-tool /
non-truncated path is byte-for-byte unchanged unless
``--tool-call-truncation-notice`` is passed.
"""

import platform
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="server import requires Apple Silicon deps",
)
class TestTruncatedToolCallNotice:
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

    def test_model_agnostic_hermes_marker(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()('<tool_call>{"name"', "length", False) is not None
        finally:
            s._tool_call_truncation_notice = False

    def test_model_agnostic_qwen3_xml_marker(self):
        """Qwen3.5/3.8 XML format: <tool_call><function=Write><parameter=..>."""
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            truncated = (
                "<tool_call>\n<function=Write>\n<parameter=file_path>/tmp/x.py"
                "</parameter>\n<parameter=content>def f():"
            )
            assert self._fn()(truncated, "length", False) is not None
        finally:
            s._tool_call_truncation_notice = False
