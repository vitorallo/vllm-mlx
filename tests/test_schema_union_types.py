# SPDX-License-Identifier: Apache-2.0
"""JSON Schema union types must not reach a chat template as a list.

Claude Code sends optional tool parameters as ``"type": ["boolean", "null"]``.
Templates that render the schema by string concatenation — Qwen3.5's among
them — raise ``TypeError: can only concatenate str (not "list") to str``
inside Jinja, which surfaces as an opaque HTTP 500 on every request carrying
such a tool.
"""

from vllm_mlx.api.tool_calling import (
    _scalarize_schema_types,
    convert_tools_for_template,
)


def test_union_collapses_to_first_non_null():
    assert _scalarize_schema_types({"type": ["boolean", "null"]})["type"] == "boolean"
    assert _scalarize_schema_types({"type": ["null", "string"]})["type"] == "string"


def test_all_null_union_still_yields_a_string():
    assert _scalarize_schema_types({"type": ["null"]})["type"] == "null"


def test_empty_union_falls_back():
    assert _scalarize_schema_types({"type": []})["type"] == "string"


def test_scalar_types_untouched():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert _scalarize_schema_types(schema) == schema


def test_nested_and_array_schemas():
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": ["integer", "null"]}},
            "nested": {
                "type": ["object", "null"],
                "properties": {"deep": {"type": ["number", "null"]}},
            },
        },
    }
    out = _scalarize_schema_types(schema)
    assert out["properties"]["items"]["items"]["type"] == "integer"
    assert out["properties"]["nested"]["type"] == "object"
    assert out["properties"]["nested"]["properties"]["deep"]["type"] == "number"


def test_input_is_not_mutated():
    schema = {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}
    _scalarize_schema_types(schema)
    assert schema["properties"]["a"]["type"] == ["string", "null"]


def test_convert_tools_for_template_scalarizes():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "description": "Edit a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "replace_all": {"type": ["boolean", "null"]},
                    },
                },
            },
        }
    ]
    out = convert_tools_for_template(tools)
    props = out[0]["function"]["parameters"]["properties"]
    assert props["replace_all"]["type"] == "boolean"
    assert props["file_path"]["type"] == "string"
