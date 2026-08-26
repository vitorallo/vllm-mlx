# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's config must not be able to drop its own stop token.

Some Qwen3 derivatives ship generation_config.json with eos_token_id 248044
(<|endoftext|>) while the tokenizer and chat template use <|im_end|> (248046).
Generation then never sees a stop token, runs past the turn boundary, and the
model hallucinates the rest of the conversation.
"""

from vllm_mlx.utils.tokenizer import _ensure_tokenizer_eos


class _Tok:
    """mlx-lm shape: a set in eos_token_ids, extended via add_eos_token."""

    def __init__(self, own, known):
        self.eos_token_id = own
        self.eos_token_ids = set(known)
        self.eos_token = "<|im_end|>"
        self.added = []

    def add_eos_token(self, tid):
        self.added.append(tid)
        self.eos_token_ids.add(tid)


class _Criteria:
    def __init__(self, ids):
        self.eos_token_ids = list(ids)
        self.added = []

    def add_eos_token_ids(self, tid):
        self.added.append(tid)
        self.eos_token_ids.append(tid)


class _VlmTok:
    """mlx-vlm shape: a StoppingCriteria seeded from the (wrong) model config,
    plus a bare int eos_token_ids on the raw HF tokenizer."""

    def __init__(self, own, criteria_ids, bare_ids=None):
        self.eos_token_id = own
        self.eos_token = "<|im_end|>"
        self.stopping_criteria = _Criteria(criteria_ids)
        if bare_ids is not None:
            self.eos_token_ids = bare_ids


def test_missing_own_eos_is_added():
    tk = _Tok(own=248046, known={248044})
    _ensure_tokenizer_eos(tk, "m")
    assert tk.added == [248046]
    assert tk.eos_token_ids == {248044, 248046}


def test_already_present_is_left_alone():
    tk = _Tok(own=248046, known={248044, 248046})
    _ensure_tokenizer_eos(tk, "m")
    assert tk.added == []


def test_union_never_drops_the_config_value():
    tk = _Tok(own=248046, known={248044})
    _ensure_tokenizer_eos(tk, "m")
    assert 248044 in tk.eos_token_ids


def test_missing_attributes_are_tolerated():
    class Bare:
        pass

    _ensure_tokenizer_eos(Bare(), "m")  # must not raise


def test_none_own_eos_is_noop():
    tk = _Tok(own=None, known={248044})
    _ensure_tokenizer_eos(tk, "m")
    assert tk.added == []


def test_add_failure_does_not_propagate():
    class Boom(_Tok):
        def add_eos_token(self, tid):
            raise RuntimeError("nope")

    _ensure_tokenizer_eos(Boom(own=248046, known={248044}), "m")  # must not raise


def test_vlm_stopping_criteria_gets_the_missing_eos():
    """The container generation actually consults on the mlx-vlm path."""
    tk = _VlmTok(own=248046, criteria_ids=[248044])
    _ensure_tokenizer_eos(tk, "m")
    assert tk.stopping_criteria.added == [248046]
    assert tk.stopping_criteria.eos_token_ids == [248044, 248046]


def test_vlm_already_correct_is_untouched():
    tk = _VlmTok(own=248046, criteria_ids=[248044, 248046])
    _ensure_tokenizer_eos(tk, "m")
    assert tk.stopping_criteria.added == []


def test_bare_int_eos_token_ids_does_not_raise():
    """A raw HF tokenizer exposes eos_token_ids as an int, not a collection.

    An earlier version did ``own in known`` against that int, raised TypeError,
    swallowed it, and silently did nothing.
    """
    tk = _VlmTok(own=248046, criteria_ids=[248044], bare_ids=248046)
    _ensure_tokenizer_eos(tk, "m")
    assert tk.stopping_criteria.eos_token_ids == [248044, 248046]


def test_criteria_failure_does_not_propagate():
    class Boom(_Criteria):
        def add_eos_token_ids(self, tid):
            raise RuntimeError("nope")

    tk = _VlmTok(own=248046, criteria_ids=[248044])
    tk.stopping_criteria = Boom([248044])
    _ensure_tokenizer_eos(tk, "m")  # must not raise
