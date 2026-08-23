"""Rendering helpers shared by every report: integers, money, aligned tables."""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(value):
    """Thousands-grouped integer: ``12000`` -> ``'12,000'``."""
    return "{:,d}".format(int(value))


def format_money(value):
    """Dollar amount to four decimal places: ``0.06`` -> ``'$0.0600'``.

    Four places, not two, because per-call costs for cheap models round to
    zero cents while still being real money at any volume.
    """
    return "${:.4f}".format(value)


def render_table(headers, rows):
    """Render ``headers`` and ``rows`` as an aligned, fixed-width text table.

    The first column is left-justified for names; every other column is
    right-justified for numbers and money, which is the shape of every table
    this package prints. A row shorter than ``headers`` pads with blanks
    rather than raising, so a caller can omit a value (a ``$/call`` on a
    TOTAL row, say) without building a full-width placeholder.
    """
    headers = [str(header) for header in headers]
    width = len(headers)
    str_rows = [[str(cell) for cell in row] for row in rows]

    widths = [len(header) for header in headers]
    for row in str_rows:
        for index in range(width):
            cell = row[index] if index < len(row) else ""
            widths[index] = max(widths[index], len(cell))

    def render_row(cells):
        parts = []
        for index in range(width):
            cell = cells[index] if index < len(cells) else ""
            justified = cell.ljust if index == 0 else cell.rjust
            parts.append(justified(widths[index]))
        return "  ".join(parts).rstrip()

    lines = [render_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(render_row(row) for row in str_rows)
    return "\n".join(lines)
