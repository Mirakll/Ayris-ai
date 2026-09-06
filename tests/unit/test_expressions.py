"""Задача 32: вычислитель выражений и подстановка ``{var}`` — что считает и что отказывается.

The expression engine is the only place in Ayris where text a stranger wrote decides what
happens next: section 7.1 lets a command carry a condition, task 36 imports conditions
wholesale out of VoiceAttack profiles, and a plugin may build one from a string it took off
the network. So this file is written in two halves of very different weight.

*It has to compute the right thing.* :class:`TestArithmetic`, :class:`TestComparisons`,
:class:`TestLogic`, :class:`TestIndexing` and :class:`TestFunctions` cover what a condition
is made of, including the one section 22 spells out — ``{work_monitor_brightness} > 0``.

*It has to refuse everything else, by construction and not by a blacklist.*
:class:`TestHostileExpressions` is the important class in this file. Attribute access,
imports, comprehensions, f-strings, walrus, lambda, ``10 ** 10 ** 10``, a ten-megabyte
string, a thousand nested parentheses, and a slot whose *value* is written to break out of
the expression it sits in — every one of them has to come back as
:class:`~ayris.actions.macros.errors.MacroExpressionError` and nothing else. A test here
that starts passing for the wrong reason (a ``SyntaxError`` escaping, say) is a hole.

Groups:

* :class:`TestTruthy` — what counts as "yes" for ``If`` and ``While``.
* :class:`TestFormatValue` — a value on its way into text: no ``50.0`` tails.
* :class:`TestCoercion` — the six declared types and their failures.
* :class:`TestSubstitute` — ``{var}`` in strings, ``{{`` escaping, the whole-string case.
* :class:`TestArithmetic` — the seven operators, integer and fractional.
* :class:`TestComparisons` — numbers, strings, mixed, chained, ``in``.
* :class:`TestLogic` — ``and``, ``or``, ``not``, short circuit, ``a if c else b``.
* :class:`TestIndexing` — arrays, dictionaries, literals, and the errors.
* :class:`TestFunctions` — the nine allowed calls, their arity, and the refusals.
* :class:`TestHostileExpressions` — every way in, closed.
* :class:`TestLimits` — text, depth, exponent, repetition.
"""

from __future__ import annotations

from typing import Any

import pytest

from ayris.actions.macros.errors import (
    MacroExpressionError,
    MacroReferenceError,
    MacroValueError,
)
from ayris.actions.macros.expressions import (
    FUNCTIONS,
    MAX_DEPTH,
    MAX_EXPRESSION,
    coerce_value,
    empty_value,
    evaluate_expression,
    format_value,
    substitute,
    truthy,
)
from ayris.core.models import VariableType

pytestmark = pytest.mark.unit


VARIABLES: dict[str, Any] = {
    "volume": 50,
    "ratio": 2.5,
    "name": "Ayris",
    "on": True,
    "off": False,
    "nothing": None,
    "work_monitor_brightness": 70,
    "items": [10, 20, 30],
    "config": {"app": "code", "monitor": "external_1"},
}


def resolve(name: str) -> Any:
    """The lookup an expression is given, failing the way the context's does.

    Raises:
        MacroReferenceError: nothing of that name exists — the same error the real
            :class:`~ayris.actions.macros.context.ExecutionContext` raises, because the
            message a user sees must not depend on who did the looking up.
    """
    if name in VARIABLES:
        return VARIABLES[name]
    raise MacroReferenceError(name)


def value_of(expression: str) -> Any:
    """One expression computed against :data:`VARIABLES`."""
    return evaluate_expression(expression, resolve)


def refused(expression: str) -> str:
    """The technical text of the refusal ``expression`` earns.

    Every hostile case goes through here rather than asserting on its own, so a case that
    starts raising something *other* than :class:`MacroExpressionError` — a bare
    ``SyntaxError``, a ``RecursionError``, an ``AttributeError`` from deep inside — fails
    instead of passing as "it raised something".
    """
    with pytest.raises(MacroExpressionError) as caught:
        evaluate_expression(expression, resolve)
    return caught.value.technical


class TestTruthy:
    """What ``If`` and ``While`` read as "yes"."""

    @pytest.mark.parametrize(
        "value",
        [True, 1, -1, 0.5, "да", "true", "текст", [0], {"a": 1}, "0.0"],
    )
    def test_yes(self, value) -> None:
        assert truthy(value) is True

    @pytest.mark.parametrize(
        "value",
        [False, 0, 0.0, "", " ", "0", "false", "FALSE", "none", "нет", "выкл", None, [], {}],
    )
    def test_no(self, value) -> None:
        assert truthy(value) is False

    def test_the_word_false_is_not_a_non_empty_string(self) -> None:
        """``bool("false")`` is ``True``, and a ``While`` reading it would spin forever."""
        assert truthy("false") is False
        assert bool("false") is True


class TestFormatValue:
    """A value on its way into text. The tails are the point."""

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (50, "50"),
            (50.0, "50"),
            (2.5, "2.5"),
            (True, "true"),
            (False, "false"),
            (None, ""),
            ("уже текст", "уже текст"),
        ],
    )
    def test_text_of(self, value, text) -> None:
        assert format_value(value) == text

    def test_containers_go_as_json_with_live_letters(self) -> None:
        assert format_value(["раз", 2]) == '["раз", 2]'
        assert format_value({"ключ": 1}) == '{"ключ": 1}'


class TestCoercion:
    """The six declared types: what fits, what is made to fit, what is refused."""

    @pytest.mark.parametrize(
        ("kind", "given", "wanted"),
        [
            (VariableType.STRING, 50, "50"),
            (VariableType.STRING, 50.0, "50"),
            (VariableType.INT, "50", 50),
            (VariableType.INT, 50.0, 50),
            (VariableType.INT, True, 1),
            (VariableType.FLOAT, "2,5", 2.5),
            (VariableType.FLOAT, 2, 2.0),
            (VariableType.BOOL, "нет", False),
            (VariableType.BOOL, "да", True),
            (VariableType.ARRAY, "[1, 2]", [1, 2]),
            (VariableType.ARRAY, (1, 2), [1, 2]),
            (VariableType.DICT, '{"a": 1}', {"a": 1}),
        ],
    )
    def test_fits(self, kind, given, wanted) -> None:
        assert coerce_value("x", given, kind) == wanted

    @pytest.mark.parametrize(
        ("kind", "given"),
        [
            (VariableType.INT, "громко"),
            (VariableType.FLOAT, "не число"),
            (VariableType.ARRAY, "не json"),
            (VariableType.ARRAY, 5),
            (VariableType.DICT, "[1, 2]"),
        ],
    )
    def test_refused_by_name(self, kind, given) -> None:
        with pytest.raises(MacroValueError) as caught:
            coerce_value("уровень", given, kind)
        assert "уровень" in str(caught.value)

    @pytest.mark.parametrize(
        ("given", "wanted"),
        [("2.5", 3), ("2.4", 2), (2.5, 3), (3.5, 4), (-2.5, -3), ("2,5", 3)],
    )
    def test_a_fractional_value_in_an_int_variable_rounds_the_way_a_person_means(
        self, given, wanted
    ) -> None:
        """Half away from zero, not Python's half-to-even: ``round(2.5)`` is 2 and 2 is wrong.

        A declared ``int`` is forgiving on purpose — a slot filled by speech arrives as
        text, and «два с половиной» in an ``int`` variable means 3 to the person who said
        it. What is refused is text that is not a number at all.
        """
        assert coerce_value("x", given, VariableType.INT) == wanted

    @pytest.mark.parametrize(
        ("kind", "empty"),
        [
            (VariableType.STRING, ""),
            (VariableType.INT, 0),
            (VariableType.FLOAT, 0.0),
            (VariableType.BOOL, False),
            (VariableType.ARRAY, []),
            (VariableType.DICT, {}),
        ],
    )
    def test_empty_of_every_type(self, kind, empty) -> None:
        assert empty_value(kind) == empty


class TestSubstitute:
    """``{var}`` inside a string parameter."""

    def test_a_value_is_written_into_text(self) -> None:
        assert substitute("громкость {volume} процентов", resolve) == "громкость 50 процентов"

    def test_a_string_that_is_one_placeholder_gives_back_the_value(self) -> None:
        """``"level": "{volume}"`` has to reach ``SetVolume`` as the number 50."""
        assert substitute("{volume}", resolve) == 50
        assert substitute("  {items}  ", resolve) == [10, 20, 30]

    def test_no_float_tail_inside_text(self) -> None:
        assert substitute("уровень {ratio}", resolve) == "уровень 2.5"
        assert substitute("{volume} и {on}", resolve) == "50 и true"

    def test_a_doubled_brace_is_a_literal_one(self) -> None:
        assert substitute("{{volume}}", resolve) == "{volume}"
        assert substitute("{{ {volume} }}", resolve) == "{ 50 }"

    def test_text_without_placeholders_is_itself(self) -> None:
        assert substitute("ничего не подставляется", resolve) == "ничего не подставляется"

    def test_an_unknown_name_says_which(self) -> None:
        with pytest.raises(MacroReferenceError) as caught:
            substitute("{нет_такой}", resolve)
        assert "нет_такой" in str(caught.value)

    def test_none_becomes_nothing_and_not_the_word_none(self) -> None:
        assert substitute("[{nothing}]", resolve) == "[]"


class TestArithmetic:
    """The seven operators, on the two kinds of number."""

    @pytest.mark.parametrize(
        ("expression", "wanted"),
        [
            ("2 + 3", 5),
            ("10 - 4", 6),
            ("6 * 7", 42),
            ("7 / 2", 3.5),
            ("7 // 2", 3),
            ("7 % 3", 1),
            ("2 ** 8", 256),
            ("-{volume}", -50),
            ("+{volume}", 50),
            ("{volume} + 10", 60),
            ("{volume} * {ratio}", 125.0),
            ("({volume} + 10) * 2", 120),
            ("{volume} / 4", 12.5),
        ],
    )
    def test_computes(self, expression, wanted) -> None:
        assert value_of(expression) == wanted

    def test_a_string_that_holds_a_number_is_one(self) -> None:
        """A slot arrives as text and still has to add up."""
        assert evaluate_expression("{level} + 1", {"level": "49"}.__getitem__) == 50

    def test_strings_concatenate_and_repeat_within_reason(self) -> None:
        assert value_of("{name} + '!'") == "Ayris!"
        assert value_of("'-' * 3") == "---"

    def test_division_by_zero_is_a_refusal_and_not_a_crash(self) -> None:
        assert "division" in refused("1 / 0").lower() or "zero" in refused("1 / 0").lower()

    def test_a_word_is_not_a_number(self) -> None:
        assert refused("{name} - 1")


class TestComparisons:
    """What a condition mostly is."""

    @pytest.mark.parametrize(
        ("expression", "wanted"),
        [
            ("{work_monitor_brightness} > 0", True),
            ("{volume} >= 50", True),
            ("{volume} > 50", False),
            ("{volume} == 50", True),
            ("{volume} != 50", False),
            ("{volume} < 100", True),
            ("{volume} <= 49", False),
            ("{name} == 'Ayris'", True),
            ("{name} != 'Ayris'", False),
            ("0 < {volume} < 100", True),
            ("0 < {volume} < 10", False),
            ("20 in {items}", True),
            ("99 in {items}", False),
            ("'app' in {config}", True),
            ("99 not in {items}", True),
        ],
    )
    def test_compares(self, expression, wanted) -> None:
        assert value_of(expression) is wanted

    def test_a_numeric_string_compares_as_a_number(self) -> None:
        assert evaluate_expression("{level} > 40", {"level": "50"}.__getitem__) is True

    def test_a_word_against_a_number_is_a_refusal_and_not_a_random_answer(self) -> None:
        """Python 2 would have answered this; a wrong branch is worse than an error."""
        assert refused("{name} > 5")

    def test_true_equals_true_and_not_one(self) -> None:
        assert value_of("{on} == true") is True
        assert value_of("{off} == false") is True


class TestLogic:
    """``and``, ``or``, ``not``, and the conditional expression."""

    @pytest.mark.parametrize(
        ("expression", "wanted"),
        [
            ("{on} and {volume} > 10", True),
            ("{off} and {volume} > 10", False),
            ("{off} or {volume} > 10", True),
            ("not {off}", True),
            ("not {on}", False),
            ("{on} and not {off}", True),
            ("({volume} > 10) and ({volume} < 100)", True),
            ("{volume} > 90 or {name} == 'Ayris'", True),
        ],
    )
    def test_computes(self, expression, wanted) -> None:
        assert value_of(expression) is wanted

    def test_a_string_is_read_the_way_if_reads_it(self) -> None:
        """``and`` gives back the operand, as Python does, and «нет» is a no.

        The operand and not a bool, so ``{a} or {b}`` can pick a value; what makes it a
        condition is :func:`truthy`, which is what ``If`` and ``While`` put it through.
        """
        assert truthy(evaluate_expression("{flag} and true", {"flag": "нет"}.__getitem__)) is False
        assert truthy(evaluate_expression("{flag} or false", {"flag": "да"}.__getitem__)) is True

    def test_or_does_not_evaluate_what_it_does_not_need(self) -> None:
        """Short circuit, and it is load-bearing: the right half would refuse."""
        assert value_of("{volume} == 50 or {name} > 5") is True

    def test_and_does_not_evaluate_what_it_does_not_need(self) -> None:
        assert value_of("{volume} == 1 and {name} > 5") is False

    def test_a_conditional_expression_picks_one_side(self) -> None:
        assert value_of("100 if {volume} > 10 else 0") == 100
        assert value_of("100 if {volume} > 90 else 0") == 0


class TestIndexing:
    """Reaching into an array or a dictionary from a condition."""

    @pytest.mark.parametrize(
        ("expression", "wanted"),
        [
            ("{items}[0]", 10),
            ("{items}[2]", 30),
            ("{items}[-1]", 30),
            ("{config}['app']", "code"),
            ("{items}[0] + {items}[1]", 30),
            ("[1, 2, 3][1]", 2),
            ("{'a': 1}['a']", 1),
            ("len({items})", 3),
        ],
    )
    def test_reads(self, expression, wanted) -> None:
        assert value_of(expression) == wanted

    def test_an_index_outside_the_array_says_so(self) -> None:
        assert "index" in refused("{items}[9]").lower()

    def test_a_missing_key_says_so(self) -> None:
        assert "key" in refused("{config}['нет']").lower()

    def test_a_slice_is_not_indexing(self) -> None:
        """No slices: nothing in section 7 needs one, and every unused door stays shut."""
        assert refused("{items}[0:2]")

    def test_a_number_cannot_be_indexed(self) -> None:
        assert refused("{volume}[0]")


class TestFunctions:
    """The nine calls of section 7.1, and only those nine."""

    def test_the_table_is_exactly_the_nine_the_specification_lists(self) -> None:
        assert set(FUNCTIONS) == {
            "len",
            "int",
            "float",
            "str",
            "round",
            "abs",
            "min",
            "max",
            "contains",
        }

    @pytest.mark.parametrize(
        ("expression", "wanted"),
        [
            ("len({items})", 3),
            ("len({name})", 5),
            ("len({config})", 2),
            ("int('50')", 50),
            ("int({ratio})", 2),
            ("int(-{ratio})", -2),
            ("float('2.5')", 2.5),
            ("float({volume})", 50.0),
            ("str({volume})", "50"),
            ("str({ratio})", "2.5"),
            ("round({ratio})", 3),
            ("round(3.5)", 4),
            ("round(-2.5)", -3),
            ("round(2.345, 2)", 2.35),
            ("round(2.675, 2)", 2.68),
            ("abs(-{volume})", 50),
            ("min({items})", 10),
            ("max({items})", 30),
            ("min(5, 2, 9)", 2),
            ("max(5, 2, 9)", 9),
            ("contains({items}, 20)", True),
            ("contains({items}, 99)", False),
            ("contains({name}, 'yr')", True),
            ("contains({config}, 'app')", True),
        ],
    )
    def test_computes(self, expression, wanted) -> None:
        assert value_of(expression) == wanted

    def test_str_of_a_whole_float_has_no_tail(self) -> None:
        """The same rule as :func:`format_value`: nobody says "50.0 percent"."""
        assert value_of("str(50.0)") == "50"

    def test_a_call_can_be_nested_in_a_comparison(self) -> None:
        assert value_of("len({items}) > 2") is True

    @pytest.mark.parametrize(
        "expression",
        ["len()", "len({items}, 2)", "round()", "round(1, 2, 3)", "contains({items})"],
    )
    def test_the_wrong_number_of_arguments_says_how_many_it_wanted(self, expression) -> None:
        assert "arguments" in refused(expression)

    def test_a_named_argument_is_refused(self) -> None:
        assert "named" in refused("round({ratio}, ndigits=2)")

    def test_a_star_argument_is_refused(self) -> None:
        assert refused("max(*{items})")

    @pytest.mark.parametrize(
        "name",
        ["print", "open", "eval", "exec", "__import__", "compile", "globals", "getattr", "type"],
    )
    def test_a_function_outside_the_table_is_refused_by_name(self, name) -> None:
        message = refused(f"{name}('os')")
        assert name in message
        assert "not allowed" in message


class TestHostileExpressions:
    """Every way out of the expression, closed. The class this module exists for.

    Each case is something a VoiceAttack profile or a plugin could contain, and every one
    of them has to come back as :class:`MacroExpressionError` — not a ``SyntaxError``, not
    an ``AttributeError``, and above all not a value.
    """

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "__import__('os').system('calc')",
            "().__class__",
            "().__class__.__base__.__subclasses__()",
            "{name}.upper()",
            "{name}.__class__.__name__",
            "{items}.append(1)",
            "{config}.keys()",
            "(1).__add__(2)",
            "''.join(['a', 'b'])",
        ],
    )
    def test_no_attribute_is_reachable(self, expression) -> None:
        """One attribute is one door to the interpreter, so there are no attributes."""
        assert refused(expression)

    @pytest.mark.parametrize(
        "expression",
        [
            "[c for c in {items}]",
            "{c for c in {items}}",
            "[c for c in ().__class__.__base__.__subclasses__()]",
            "(c for c in {items})",
            "{k: 1 for k in {items}}",
        ],
    )
    def test_no_comprehension(self, expression) -> None:
        assert refused(expression)

    @pytest.mark.parametrize(
        "expression",
        [
            "lambda: 1",
            "(lambda x: x)(1)",
            "f'{1+1}'",
            "(x := 5)",
            "...",
            "1;2",
            "import os",
            "x = 5",
            "await 1",
            "yield 1",
        ],
    )
    def test_no_statement_and_no_exotic_expression(self, expression) -> None:
        assert refused(expression)

    def test_builtins_are_not_names_in_scope(self) -> None:
        """``open`` as a bare name is a lookup, and the lookup does not know it."""
        for name in ("open", "__builtins__", "__name__", "globals"):
            with pytest.raises((MacroExpressionError, MacroReferenceError)):
                evaluate_expression(name, resolve)

    def test_a_slot_value_cannot_close_the_expression_it_sits_in(self) -> None:
        """The reason a value never becomes source: ``{x}`` is bound, not pasted.

        Were the text substituted before parsing, ``{x} == 1`` with ``x`` holding
        ``50) or (1`` would parse as ``50) or (1 == 1`` and answer ``True``. It is bound to
        a generated name instead, so the hostile text stays a string that is compared and
        found unequal.
        """
        hostile = {"x": "50) or (1"}
        assert evaluate_expression("{x} == 1", hostile.__getitem__) is False

    @pytest.mark.parametrize(
        "payload",
        [
            "__import__('os').system('calc')",
            "().__class__",
            "'; DROP TABLE commands; --",
            "{{}}",
        ],
    )
    def test_a_hostile_slot_value_is_only_ever_a_string(self, payload) -> None:
        assert evaluate_expression("{x} == 'да'", {"x": payload}.__getitem__) is False
        assert evaluate_expression("len({x}) > 0", {"x": payload}.__getitem__) is True

    def test_an_unknown_name_is_a_reference_error_and_names_itself(self) -> None:
        with pytest.raises(MacroReferenceError) as caught:
            value_of("{нет_такой} > 1")
        assert "нет_такой" in str(caught.value)

    def test_an_empty_expression_is_refused(self) -> None:
        assert refused("")
        assert refused("   ")


class TestLimits:
    """The four ceilings, each of them a way to hang a thread the user cannot see."""

    def test_text_longer_than_the_ceiling_is_refused_before_it_parses(self) -> None:
        message = refused("1 + " * MAX_EXPRESSION + "1")
        assert str(MAX_EXPRESSION) in message

    def test_a_tower_of_exponents_is_refused(self) -> None:
        """``10 ** 10 ** 10`` is nine characters and minutes of a core."""
        assert refused("10 ** 10 ** 10")
        assert refused("2 ** 1000")

    def test_a_small_exponent_still_works(self) -> None:
        assert value_of("2 ** 8") == 256

    def test_a_huge_repetition_is_refused(self) -> None:
        assert refused("'a' * 10000000")
        assert refused("[1] * 10000000")

    def test_a_small_repetition_still_works(self) -> None:
        assert value_of("'ab' * 3") == "ababab"

    def test_deep_nesting_is_refused_and_not_a_recursion_error(self) -> None:
        """Depth is counted in nodes, not in characters: ``((1))`` is one node, not three."""
        deep = "(" * (MAX_DEPTH + 5) + "1" + " + 1)" * (MAX_DEPTH + 5)
        assert "deeply" in refused(deep)

    def test_deep_nesting_through_operators_is_refused_too(self) -> None:
        assert "deeply" in refused("-" * (MAX_DEPTH + 5) + "1")

    def test_bare_parentheses_do_not_count_as_depth(self) -> None:
        """They are not nodes at all, so a person may bracket as much as they like."""
        assert value_of("(" * 200 + "1" + ")" * 200) == 1

    def test_nesting_within_the_ceiling_works(self) -> None:
        assert value_of("((({volume} + 1)))") == 51
