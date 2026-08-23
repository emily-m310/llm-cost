"""Aggregate usage records into a cost report, and rank models by price.

Both operations share the same rule as the rest of the package: a model with
no price is never folded into a total as zero. ``build_report`` counts it in
``unknown_models`` and drops its records from every group's totals instead.
"""

from collections import namedtuple

from .estimate import estimate_cost
from .pricing import UnknownModelError

__all__ = ["GroupSummary", "Report", "ComparisonRow", "build_report", "compare_models"]

#: One row of a `compare_models` result: a model's price, what the workload
#: would cost against it, and that cost relative to the cheapest option.
ComparisonRow = namedtuple("ComparisonRow", ["price", "breakdown", "vs_cheapest"])


class GroupSummary(object):
    """Running totals for one group (a model, a team, a day) in a report."""

    __slots__ = (
        "key",
        "calls",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "cost",
    )

    def __init__(self, key):
        self.key = key
        self.calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.cost / self.calls

    def _add(self, record, cost):
        self.calls += 1
        self.input_tokens += record.input_tokens
        self.cached_input_tokens += record.cached_input_tokens
        self.cache_write_tokens += record.cache_write_tokens
        self.output_tokens += record.output_tokens
        self.cost += cost

    def to_dict(self):
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "cost": round(self.cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }

    def __repr__(self):
        return "GroupSummary(%s, calls=%d, cost=%.6f)" % (
            self.key,
            self.calls,
            self.cost,
        )


class Report(object):
    """A cost report grouped by one field of a set of usage records."""

    __slots__ = ("group_by", "groups", "unknown_models", "skipped")

    def __init__(self, group_by, groups, unknown_models, skipped):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.skipped = skipped

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_input_tokens(self):
        return sum(group.input_tokens for group in self.groups)

    @property
    def total_cached_input_tokens(self):
        return sum(group.cached_input_tokens for group in self.groups)

    @property
    def total_cache_write_tokens(self):
        return sum(group.cache_write_tokens for group in self.groups)

    @property
    def total_output_tokens(self):
        return sum(group.output_tokens for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.cost for group in self.groups)

    def to_dict(self):
        return {
            "group_by": self.group_by,
            "groups": [group.to_dict() for group in self.groups],
            "total_calls": self.total_calls,
            "total_cost": round(self.total_cost, 6),
            "unknown_models": sorted(self.unknown_models),
            "skipped": self.skipped,
        }

    def __repr__(self):
        return "Report(group_by=%s, groups=%d, total=%.6f)" % (
            self.group_by,
            len(self.groups),
            self.total_cost,
        )


def build_report(records, pricing, group_by="model"):
    """Aggregate ``records`` by ``group_by`` and cost each group against ``pricing``.

    Groups come back sorted by cost, most expensive first, so the biggest
    line item on the bill is always the first thing read.
    """
    groups = {}
    unknown_models = set()
    skipped = 0
    for record in records:
        try:
            price = pricing.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            skipped += 1
            continue
        breakdown = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )
        key = record.group_key(group_by)
        group = groups.get(key)
        if group is None:
            group = GroupSummary(key)
            groups[key] = group
        group._add(record, breakdown.total_cost)

    ordered = sorted(groups.values(), key=lambda group: group.cost, reverse=True)
    return Report(group_by, ordered, unknown_models, skipped)


def compare_models(pricing, input_tokens, output_tokens, calls=1, models=None, provider=None):
    """Cost one workload against every model in ``pricing``, cheapest first.

    ``models`` narrows the comparison to that shortlist of names (resolved
    the same way a usage log would be); ``provider`` filters the full table
    to one provider's models. Passing both is an error, since a shortlist
    already says exactly which models to compare.
    """
    if models and provider:
        raise ValueError("pass models or provider, not both")

    if models:
        prices = [pricing.resolve(name) for name in models]
    else:
        prices = [pricing.prices[name] for name in pricing.models()]
        if provider:
            prices = [price for price in prices if price.provider == provider]
    if not prices:
        raise ValueError("no models to compare")

    priced = []
    for price in prices:
        breakdown = estimate_cost(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            calls=calls,
        )
        priced.append((price, breakdown))
    priced.sort(key=lambda pair: pair[1].total_cost)

    cheapest = priced[0][1].total_cost
    return [
        ComparisonRow(
            price=price,
            breakdown=breakdown,
            vs_cheapest=1.0 if cheapest == 0 else breakdown.total_cost / cheapest,
        )
        for price, breakdown in priced
    ]
