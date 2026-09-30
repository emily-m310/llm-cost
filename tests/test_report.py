import pytest

from llm_cost.pricing import parse_pricing
from llm_cost.report import build_report, compare_models
from llm_cost.usage import load_usage

TABLE = parse_pricing(
    {
        "replace": True,
        "models": {
            "cheap": {"input": 1.0, "output": 1.0, "provider": "acme"},
            "pricey": {"input": 10.0, "output": 10.0, "provider": "acme"},
            "other": {"input": 100.0, "output": 100.0, "provider": "other-co"},
        },
    }
)


def _records(*lines):
    records, problems = load_usage("\n".join(lines))
    assert not problems
    return records


def test_build_report_sorts_groups_by_cost_descending():
    records = _records(
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
        '{"model": "pricey", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
    )
    report = build_report(records, TABLE)
    assert [group.key for group in report.groups] == ["pricey", "cheap"]
    assert report.total_cost == pytest.approx(11.0)
    assert report.total_calls == 2


def test_build_report_skips_unpriced_models_without_zeroing_total():
    records = _records(
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
        '{"model": "unknown-model", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
    )
    report = build_report(records, TABLE)
    assert report.unknown_models == {"unknown-model"}
    assert report.skipped == 1
    assert [group.key for group in report.groups] == ["cheap"]
    assert report.total_cost == pytest.approx(1.0)


def test_build_report_groups_by_arbitrary_field():
    records = _records(
        '{"model": "cheap", "team": "a", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
        '{"model": "cheap", "team": "b", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
    )
    report = build_report(records, TABLE, group_by="team")
    assert sorted(group.key for group in report.groups) == ["a", "b"]


def test_build_report_prices_cached_tokens_at_the_cached_rate():
    table = parse_pricing(
        {
            "replace": True,
            "models": {
                "cachey": {"input": 2.0, "output": 4.0, "cached_input": 0.5, "provider": "acme"},
            },
        }
    )
    records = _records(
        '{"model": "cachey", "usage": {"input_tokens": 1000000, "output_tokens": 500000,'
        ' "cache_read_input_tokens": 2000000}}',
    )
    report = build_report(records, table)
    group = report.groups[0]
    assert group.cached_input_tokens == 2000000
    # 1M fresh input at 2.00, 2M cached at 0.50, 0.5M output at 4.00
    assert group.cost == pytest.approx(5.0)
    assert report.total_cached_input_tokens == 2000000


def test_build_report_with_no_records_is_empty():
    report = build_report([], TABLE)
    assert report.groups == []
    assert report.total_cost == 0
    assert report.total_calls == 0
    assert report.skipped == 0


def test_report_to_dict_is_json_ready():
    records = _records(
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}',
        '{"model": "mystery", "usage": {"input_tokens": 5, "output_tokens": 5}}',
    )
    data = build_report(records, TABLE).to_dict()
    assert data["group_by"] == "model"
    assert data["total_calls"] == 2
    assert data["total_cost"] == pytest.approx(2.0)
    assert data["unknown_models"] == ["mystery"]
    assert data["skipped"] == 1
    assert data["groups"][0]["key"] == "cheap"
    assert data["groups"][0]["cost_per_call"] == pytest.approx(1.0)


def test_compare_models_ranks_cheapest_first():
    rows = compare_models(TABLE, input_tokens=1000000, output_tokens=0, provider="acme")
    assert [row.price.model for row in rows] == ["cheap", "pricey"]
    assert rows[0].vs_cheapest == pytest.approx(1.0)
    assert rows[1].vs_cheapest == pytest.approx(10.0)


def test_compare_models_narrows_to_shortlist():
    rows = compare_models(TABLE, input_tokens=1, output_tokens=1, models=["pricey"])
    assert [row.price.model for row in rows] == ["pricey"]


def test_compare_models_filters_by_provider():
    rows = compare_models(TABLE, input_tokens=1, output_tokens=1, provider="acme")
    assert {row.price.model for row in rows} == {"cheap", "pricey"}
    other_rows = compare_models(TABLE, input_tokens=1, output_tokens=1, provider="other-co")
    assert [row.price.model for row in other_rows] == ["other"]


def test_compare_models_rejects_a_provider_with_no_models():
    with pytest.raises(ValueError):
        compare_models(TABLE, input_tokens=1, output_tokens=1, provider="nobody")


def test_compare_models_free_workload_is_one_times_everywhere():
    rows = compare_models(TABLE, input_tokens=0, output_tokens=0)
    assert [row.vs_cheapest for row in rows] == [1.0, 1.0, 1.0]


def test_compare_models_rejects_models_and_provider_together():
    with pytest.raises(ValueError):
        compare_models(TABLE, input_tokens=1, output_tokens=1, models=["cheap"], provider="acme")
