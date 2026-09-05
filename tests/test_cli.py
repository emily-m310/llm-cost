import json

import pytest

from llm_cost.cli import main

PRICING = {
    "replace": True,
    "models": {
        "cheap": {"input": 1.0, "output": 1.0, "provider": "acme"},
        "pricey": {"input": 10.0, "output": 10.0, "provider": "acme"},
    },
}


@pytest.fixture
def pricing_file(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(PRICING), encoding="utf-8")
    return str(path)


def test_estimate_prints_table(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "estimate", "--model", "cheap", "--input", "1000000", "--output", "1000000"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cheap" in out
    assert "cost per call: $2.0000" in out


def test_estimate_json_output(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "--json", "estimate", "--model", "cheap", "--input", "1000000"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["provider"] == "acme"
    assert data["input_cost"] == 1.0


def test_estimate_unknown_model_exits_3(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "estimate", "--model", "does-not-exist"])
    err = capsys.readouterr().err
    assert code == 3
    assert "does-not-exist" in err


def test_report_prints_table_and_totals(capsys, pricing_file, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text(
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}\n'
        '{"model": "pricey", "usage": {"input_tokens": 1000000, "output_tokens": 0}}\n',
        encoding="utf-8",
    )
    code = main(["--pricing", pricing_file, "report", str(usage_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert "TOTAL" in out
    assert "pricey" in out and "cheap" in out


def test_report_json_includes_problems(capsys, pricing_file, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text(
        '{"model": "cheap", "usage": {"input_tokens": 1000000, "output_tokens": 0}}\n'
        "not json at all\n",
        encoding="utf-8",
    )
    code = main(["--pricing", pricing_file, "--json", "report", str(usage_file)])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert len(data["problems"]) == 1
    assert data["problems"][0]["line"] == 2


def test_report_strict_stops_on_malformed_line(capsys, pricing_file, tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text("not json at all\n", encoding="utf-8")
    code = main(["--pricing", pricing_file, "report", "--strict", str(usage_file)])
    err = capsys.readouterr().err
    assert code == 2
    assert "line 1" in err


def test_report_missing_file_exits_2(capsys, pricing_file, tmp_path):
    missing = str(tmp_path / "does-not-exist.jsonl")
    code = main(["--pricing", pricing_file, "report", missing])
    err = capsys.readouterr().err
    assert code == 2
    assert missing in err


def test_compare_ranks_cheapest_first(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "compare", "--input", "1000000", "--output", "0"])
    out = capsys.readouterr().out
    assert code == 0
    lines = out.strip().splitlines()
    assert lines.index([line for line in lines if line.startswith("cheap")][0]) < lines.index(
        [line for line in lines if line.startswith("pricey")][0]
    )


def test_compare_json_output(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "--json", "compare", "--input", "1", "--output", "1"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert [row["provider"] for row in data["models"]] == ["acme", "acme"]


def test_models_lists_price_table(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "cheap" in out and "pricey" in out


def test_models_json_output(capsys, pricing_file):
    code = main(["--pricing", pricing_file, "--json", "models"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert set(data["models"]) == {"cheap", "pricey"}


def test_no_subcommand_prints_help_and_exits_2(capsys):
    code = main([])
    out = capsys.readouterr().out
    assert code == 2
    assert "usage" in out.lower()


def test_bad_pricing_file_exits_2(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    code = main(["--pricing", str(bad), "models"])
    err = capsys.readouterr().err
    assert code == 2
    assert "not valid JSON" in err
