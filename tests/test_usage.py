import pytest

from llm_cost.usage import aggregate, load_usage, parse_record


def test_parse_record_anthropic_shape_excludes_cache_reads_already():
    record = parse_record(
        {
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 18400,
                "output_tokens": 2100,
                "cache_read_input_tokens": 52000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    assert record.input_tokens == 18400
    assert record.cached_input_tokens == 52000
    assert record.cache_write_tokens == 9000
    assert record.output_tokens == 2100


def test_parse_record_openai_shape_subtracts_cached_prefix():
    record = parse_record(
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 31000,
                "completion_tokens": 420,
                "prompt_tokens_details": {"cached_tokens": 24000},
            },
        }
    )
    assert record.input_tokens == 7000
    assert record.cached_input_tokens == 24000
    assert record.output_tokens == 420


def test_parse_record_flat_shape_reads_model_from_usage():
    record = parse_record({"usage": {"model": "gpt-4o", "input_tokens": 10, "output_tokens": 5}})
    assert record.model == "gpt-4o"
    assert record.input_tokens == 10


def test_parse_record_requires_model():
    with pytest.raises(ValueError):
        parse_record({"usage": {"input_tokens": 1}})


def test_parse_record_rejects_negative_tokens():
    with pytest.raises(ValueError):
        parse_record({"model": "gpt-4o", "usage": {"input_tokens": -1}})


def test_parse_record_rejects_non_numeric_tokens():
    with pytest.raises(ValueError):
        parse_record({"model": "gpt-4o", "usage": {"input_tokens": "a lot"}})


def test_load_usage_skips_blank_lines_and_collects_problems():
    text = "\n".join(
        [
            '{"model": "gpt-4o", "usage": {"input_tokens": 10, "output_tokens": 5}}',
            "",
            "not json",
            '{"usage": {"input_tokens": 1}}',
        ]
    )
    records, problems = load_usage(text)
    assert len(records) == 1
    assert len(problems) == 2
    assert problems[0].line_number == 3
    assert problems[1].line_number == 4


def test_load_usage_strict_raises_on_first_bad_line():
    text = "not json"
    with pytest.raises(ValueError):
        load_usage(text, strict=True)


def test_aggregate_groups_by_model():
    records, _ = load_usage(
        "\n".join(
            [
                '{"model": "gpt-4o", "usage": {"input_tokens": 1, "output_tokens": 1}}',
                '{"model": "gpt-4o", "usage": {"input_tokens": 1, "output_tokens": 1}}',
                '{"model": "gpt-4o-mini", "usage": {"input_tokens": 1, "output_tokens": 1}}',
            ]
        )
    )
    buckets = aggregate(records, group_by="model")
    assert len(buckets["gpt-4o"]) == 2
    assert len(buckets["gpt-4o-mini"]) == 1


def test_group_key_truncates_date_and_falls_back_for_missing_field():
    record = parse_record(
        {
            "model": "gpt-4o",
            "date": "2026-06-01T09:12:00Z",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert record.group_key("date") == "2026-06-01"
    assert record.group_key("team") == "(none)"
