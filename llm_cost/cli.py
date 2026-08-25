"""Command-line entry point: estimate, report, compare, models.

Every subcommand shares two flags -- ``--pricing`` and ``--json`` -- and
accepts them either before or after the subcommand name, so they are
declared on a parent parser reused by the top-level parser and each
subparser rather than duplicated by hand.
"""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

__all__ = ["main"]


def _call_desc(calls):
    if calls == 1:
        return "1 call"
    return "%s calls" % format_int(calls)


def _rate(value):
    return "%.2f" % value


def _load_table(args):
    if args.pricing:
        return load_pricing(args.pricing)
    return default_pricing()


def _print_json(data):
    print(json.dumps(data, indent=2))


def cmd_estimate(args):
    table = _load_table(args)
    price = table.resolve(args.model)
    breakdown = estimate_cost(
        price,
        input_tokens=args.input,
        output_tokens=args.output,
        cached_input_tokens=args.cached,
        cache_write_tokens=args.cache_write,
        calls=args.calls,
    )

    if args.json:
        data = breakdown.to_dict()
        data["provider"] = price.provider
        data["as_of"] = table.as_of
        _print_json(data)
        return 0

    headers = ["item", "tokens", "$/1M", "cost"]
    rows = [
        ["input", format_int(breakdown.input_tokens), _rate(price.input), format_money(breakdown.input_cost)],
        ["cached input", format_int(breakdown.cached_input_tokens), _rate(price.cached_input), format_money(breakdown.cached_input_cost)],
        ["cache write", format_int(breakdown.cache_write_tokens), _rate(price.cache_write), format_money(breakdown.cache_write_cost)],
        ["output", format_int(breakdown.output_tokens), _rate(price.output), format_money(breakdown.output_cost)],
        ["total", format_int(breakdown.total_tokens), "", format_money(breakdown.total_cost)],
    ]

    print("%s  (%s, prices as of %s)" % (price.model, _call_desc(args.calls), table.as_of))
    print("")
    print(render_table(headers, rows))
    print("")
    print("cost per call: %s" % format_money(breakdown.cost_per_call))
    return 0


def cmd_report(args):
    table = _load_table(args)
    try:
        with open(args.usage_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise ValueError("could not read %s: %s" % (args.usage_file, error))

    records, problems = load_usage(text, strict=args.strict)
    report = build_report(records, table, group_by=args.group_by)

    if args.json:
        data = report.to_dict()
        data["problems"] = [
            {"line": problem.line_number, "message": problem.message} for problem in problems
        ]
        _print_json(data)
        return 0

    headers = [args.group_by, "calls", "input", "cached", "output", "cost", "$/call"]
    rows = [
        [
            group.key,
            format_int(group.calls),
            format_int(group.input_tokens),
            format_int(group.cached_input_tokens),
            format_int(group.output_tokens),
            format_money(group.cost),
            format_money(group.cost_per_call),
        ]
        for group in report.groups
    ]
    rows.append(
        [
            "TOTAL",
            format_int(report.total_calls),
            format_int(report.total_input_tokens),
            format_int(report.total_cached_input_tokens),
            format_int(report.total_output_tokens),
            format_money(report.total_cost),
        ]
    )
    print(render_table(headers, rows))

    if report.skipped:
        print("")
        print(
            "skipped %d record(s) with no price: %s"
            % (report.skipped, ", ".join(sorted(report.unknown_models)))
        )
    if problems:
        print("")
        print("skipped %d malformed line(s):" % len(problems))
        for problem in problems:
            print("  line %d: %s" % (problem.line_number, problem.message))
    return 0


def cmd_compare(args):
    table = _load_table(args)
    models = [name.strip() for name in args.models.split(",")] if args.models else None
    comparison = compare_models(
        table,
        input_tokens=args.input,
        output_tokens=args.output,
        calls=args.calls,
        models=models,
        provider=args.provider,
    )

    if args.json:
        data = {
            "input_tokens": args.input,
            "output_tokens": args.output,
            "calls": args.calls,
            "as_of": table.as_of,
            "models": [
                dict(
                    row.breakdown.to_dict(),
                    provider=row.price.provider,
                    vs_cheapest=round(row.vs_cheapest, 4),
                )
                for row in comparison
            ],
        }
        _print_json(data)
        return 0

    headers = ["model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest"]
    rows = [
        [
            row.price.model,
            row.price.provider,
            _rate(row.price.input),
            _rate(row.price.output),
            format_money(row.breakdown.total_cost),
            "%.1fx" % row.vs_cheapest,
        ]
        for row in comparison
    ]

    print(
        "%s in + %s out, %s, prices as of %s"
        % (format_int(args.input), format_int(args.output), _call_desc(args.calls), table.as_of)
    )
    print("")
    print(render_table(headers, rows))
    return 0


def cmd_models(args):
    table = _load_table(args)

    if args.json:
        data = {
            "as_of": table.as_of,
            "source": table.source,
            "models": dict((name, table.prices[name].to_dict()) for name in table.models()),
        }
        _print_json(data)
        return 0

    print(
        "%d models, USD per 1M tokens, as of %s (source: %s)"
        % (len(table), table.as_of, table.source)
    )
    print("")
    headers = ["model", "provider", "input", "output", "cached input", "cache write"]
    rows = [
        [
            name,
            table.prices[name].provider,
            _rate(table.prices[name].input),
            _rate(table.prices[name].output),
            _rate(table.prices[name].cached_input),
            _rate(table.prices[name].cache_write),
        ]
        for name in table.models()
    ]
    print(render_table(headers, rows))
    return 0


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pricing", metavar="FILE", help="pricing override file (see README)")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    parser = argparse.ArgumentParser(prog="llm-cost", parents=[common])
    subparsers = parser.add_subparsers(dest="command")

    estimate = subparsers.add_parser(
        "estimate", parents=[common], help="cost of one call or a batch of identical calls"
    )
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--input", type=int, default=0, help="input tokens per call")
    estimate.add_argument("--output", type=int, default=0, help="output tokens per call")
    estimate.add_argument("--cached", type=int, default=0, help="cached input tokens per call")
    estimate.add_argument("--cache-write", type=int, default=0, help="cache write tokens per call")
    estimate.add_argument("--calls", type=int, default=1)
    estimate.set_defaults(func=cmd_estimate)

    report = subparsers.add_parser(
        "report", parents=[common], help="cost report from a JSONL usage log"
    )
    report.add_argument("usage_file")
    report.add_argument("--group-by", default="model")
    report.add_argument(
        "--strict", action="store_true", help="fail on the first malformed line instead of skipping it"
    )
    report.set_defaults(func=cmd_report)

    compare = subparsers.add_parser(
        "compare", parents=[common], help="rank models cheapest first for one workload"
    )
    compare.add_argument("--input", type=int, default=0, help="input tokens per call")
    compare.add_argument("--output", type=int, default=0, help="output tokens per call")
    compare.add_argument("--calls", type=int, default=1)
    compare.add_argument("--provider", help="restrict to one provider")
    compare.add_argument("--models", help="comma-separated shortlist of model names")
    compare.set_defaults(func=cmd_compare)

    models = subparsers.add_parser("models", parents=[common], help="list the price table")
    models.set_defaults(func=cmd_models)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    try:
        return args.func(args)
    except UnknownModelError as error:
        print("error: %s" % error, file=sys.stderr)
        return 3
    except ValueError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
