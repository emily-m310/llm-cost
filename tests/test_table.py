from llm_cost.table import format_int, format_money, render_table


def test_format_int_groups_thousands():
    assert format_int(1234567) == "1,234,567"
    assert format_int(0) == "0"
    assert format_int(12) == "12"


def test_format_money_uses_four_places():
    assert format_money(0.06) == "$0.0600"
    assert format_money(1) == "$1.0000"
    assert format_money(0) == "$0.0000"


def test_render_table_aligns_columns():
    headers = ["model", "cost"]
    rows = [["claude-opus-5", "$0.0800"], ["gpt-4o-mini", "$0.0031"]]
    text = render_table(headers, rows)
    lines = text.splitlines()

    # widths are the longest of header and every cell in that column
    width0, width1 = 13, 7  # len("claude-opus-5"), len("$0.0800")
    assert lines[0] == "  ".join(["model".ljust(width0), "cost".rjust(width1)])
    assert lines[1] == "  ".join(["-" * width0, "-" * width1])
    assert lines[2] == "  ".join(["claude-opus-5".ljust(width0), "$0.0800".rjust(width1)])
    assert lines[3] == "  ".join(["gpt-4o-mini".ljust(width0), "$0.0031".rjust(width1)])


def test_render_table_pads_short_rows():
    headers = ["item", "tokens", "cost"]
    rows = [["total", "12,800", ""]]
    text = render_table(headers, rows)
    # the trailing blank cell must not raise and the row must not grow columns
    assert "total" in text.splitlines()[-1]
