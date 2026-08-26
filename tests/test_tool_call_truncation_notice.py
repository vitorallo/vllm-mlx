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


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = _FakeFunction(name, arguments)


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

    def test_empty_tool_call_shell_on_truncation_fires(self):
        """Qwen3 XML parser recognises <function=Write> then runs out of tokens.

        It emits a tool call whose arguments never arrived. Executing that is
        worse than nothing, so the notice must fire.
        """
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            shell = [_FakeToolCall("Write", "{}")]
            assert self._fn()("<tool_call><function=Write>", "length", shell) is not None
        finally:
            s._tool_call_truncation_notice = False

    def test_usable_tool_call_on_truncation_does_not_fire(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            real = [_FakeToolCall("Write", '{"file_path": "/tmp/x"}')]
            assert self._fn()("<tool_call><function=Write>", "length", real) is None
        finally:
            s._tool_call_truncation_notice = False

    def test_truncated_json_arguments_fire(self):
        """The dangerous shape: arguments cut off mid-string, JSON never closes."""
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            partial = [
                _FakeToolCall("Write", '{"file_path": "/tmp/x.py", "content": "def f(')
            ]
            assert self._fn()("<tool_call><function=Write>", "length", partial) is not None
        finally:
            s._tool_call_truncation_notice = False

    def test_unknown_tool_call_shape_stays_conservative(self):
        import vllm_mlx.server as s

        s._tool_call_truncation_notice = True
        try:
            assert self._fn()("<tool_call>", "length", [object()]) is None
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
