"""Read usage logs written in OpenAI or Anthropic shape.

The one thing worth getting right here is what ``input_tokens`` means:

* OpenAI reports ``usage.prompt_tokens`` **including** the cached prefix, with
  the cached part broken out in ``prompt_tokens_details.cached_tokens``.
* Anthropic reports ``usage.input_tokens`` **excluding** cache reads, which are
  counted separately in ``cache_read_input_tokens``.

Pricing an OpenAI record without subtracting the cached tokens charges the
cached prefix twice, at the full input rate, which overstates the bill of a
heavily cached agent by a wide margin. Records are normalised to the exclusive
form on the way in.
"""

import json

__all__ = [
    "UsageRecord",
    "UsageProblem",
    "parse_record",
    "load_usage",
    "aggregate",
]


class UsageProblem(object):
    """A line that could not be read, kept so the report can mention it."""

    __slots__ = ("line_number", "message")

    def __init__(self, line_number, message):
        self.line_number = line_number
        self.message = message

    def __repr__(self):
        return "UsageProblem(line=%d, %s)" % (self.line_number, self.message)


class UsageRecord(object):
    """One normalised API call."""

    __slots__ = (
        "model",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "fields",
    )

    def __init__(
        self,
        model,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_tokens=0,
        fields=None,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.fields = fields or {}

    def group_key(self, group_by):
        """Value used to bucket this record in a report."""
        if group_by == "model":
            return self.model
        value = self.fields.get(group_by)
        if value is None:
            return "(none)"
        if group_by == "date" and isinstance(value, str):
            return value[:10]
        return str(value)

    def __repr__(self):
        return "UsageRecord(%s, in=%d, out=%d, cached=%d)" % (
            self.model,
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
        )


def _as_int(value, label):
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number, got %r" % (label, value))
    if value < 0:
        raise ValueError("%s must not be negative" % label)
    return int(value)


def parse_record(entry):
    """Normalise one decoded usage object into a :class:`UsageRecord`."""
    if not isinstance(entry, dict):
        raise ValueError("usage record must be a JSON object")

    model = entry.get("model")
    usage = entry.get("usage")
    if isinstance(usage, dict):
        if model is None:
            model = usage.get("model")
    else:
        usage = entry
    if not model:
        raise ValueError("usage record has no model")

    cached = _as_int(
        usage.get("cache_read_input_tokens"), "cache_read_input_tokens"
    )
    cache_write = _as_int(
        usage.get("cache_creation_input_tokens"), "cache_creation_input_tokens"
    )
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached += _as_int(details.get("cached_tokens"), "cached_tokens")

    if "prompt_tokens" in usage:
        # OpenAI shape: prompt_tokens already contains the cached prefix.
        prompt = _as_int(usage.get("prompt_tokens"), "prompt_tokens")
        input_tokens = max(0, prompt - cached)
        output_tokens = _as_int(usage.get("completion_tokens"), "completion_tokens")
    else:
        # Anthropic and flat shapes: input_tokens excludes cache reads.
        input_tokens = _as_int(usage.get("input_tokens"), "input_tokens")
        output_tokens = _as_int(usage.get("output_tokens"), "output_tokens")

    fields = dict((key, value) for key, value in entry.items() if key != "usage")
    return UsageRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=cache_write,
        fields=fields,
    )


def load_usage(text, strict=False):
    """Parse JSONL ``text`` into ``(records, problems)``.

    Blank lines are ignored. A malformed line is collected as a
    :class:`UsageProblem` so one bad row does not lose the whole report; with
    ``strict`` it raises instead.
    """
    records = []
    problems = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except ValueError as error:
            problem = UsageProblem(line_number, "invalid JSON: %s" % error)
            if strict:
                raise ValueError("line %d: %s" % (line_number, problem.message))
            problems.append(problem)
            continue
        try:
            records.append(parse_record(entry))
        except ValueError as error:
            if strict:
                raise ValueError("line %d: %s" % (line_number, error))
            problems.append(UsageProblem(line_number, str(error)))
    return records, problems


def aggregate(records, group_by="model"):
    """Bucket records by ``group_by``, returning ``{key: [records]}``."""
    buckets = {}
    for record in records:
        buckets.setdefault(record.group_key(group_by), []).append(record)
    return buckets
