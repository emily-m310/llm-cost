import pytest

from llm_cost.pricing import (
    ModelPrice,
    UnknownModelError,
    default_pricing,
    load_pricing,
    parse_pricing,
)


def test_model_price_defaults_cache_fields_to_input_price():
    price = ModelPrice("m", input=2.0, output=8.0)
    assert price.cached_input == 2.0
    assert price.cache_write == 2.0


def test_model_price_rejects_negative_prices():
    with pytest.raises(ValueError):
        ModelPrice("m", input=-1.0, output=8.0)


def test_unknown_model_error_str_has_no_repr_quoting():
    error = UnknownModelError("no price for model 'x'")
    assert str(error) == "no price for model 'x'"


def test_resolve_exact_and_case_insensitive():
    table = default_pricing()
    assert table.resolve("claude-opus-5").model == "claude-opus-5"
    assert table.resolve("CLAUDE-OPUS-5").model == "claude-opus-5"


def test_resolve_strips_provider_prefix():
    table = default_pricing()
    assert table.resolve("anthropic.claude-opus-5").model == "claude-opus-5"
    assert table.resolve("openai/gpt-4o-mini").model == "gpt-4o-mini"


def test_resolve_matches_longest_prefix_for_date_suffixes():
    table = default_pricing()
    assert table.resolve("gpt-4o-mini-2026-01-31").model == "gpt-4o-mini"
    assert table.resolve("gpt-4o-2026-01-31").model == "gpt-4o"


def test_resolve_raises_on_unknown_model():
    table = default_pricing()
    with pytest.raises(UnknownModelError):
        table.resolve("internal-router-v3")
    with pytest.raises(UnknownModelError):
        table.resolve("")


def test_parse_pricing_merges_onto_builtin_by_default():
    table = parse_pricing({"my-model": {"input": 1, "output": 3}})
    assert table.resolve("my-model").input == 1.0
    assert table.resolve("claude-opus-5").input == 5.0


def test_parse_pricing_replace_drops_builtin():
    table = parse_pricing({"replace": True, "models": {"my-model": {"input": 1, "output": 3}}})
    assert table.resolve("my-model").input == 1.0
    with pytest.raises(UnknownModelError):
        table.resolve("claude-opus-5")


def test_parse_pricing_rejects_missing_fields():
    with pytest.raises(ValueError):
        parse_pricing({"my-model": {"input": 1}})


def test_parse_pricing_rejects_non_numeric_field():
    with pytest.raises(ValueError):
        parse_pricing({"my-model": {"input": 1, "output": "a lot"}})


def test_parse_pricing_rejects_empty_models():
    with pytest.raises(ValueError):
        parse_pricing({"models": {}})


def test_load_pricing_missing_file(tmp_path):
    with pytest.raises(ValueError):
        load_pricing(str(tmp_path / "does-not-exist.json"))


def test_load_pricing_invalid_json(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pricing(str(path))


def test_load_pricing_reads_file(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"my-model": {"input": 1, "output": 3}}', encoding="utf-8")
    table = load_pricing(str(path))
    assert table.resolve("my-model").output == 3.0
    assert table.source == str(path)
