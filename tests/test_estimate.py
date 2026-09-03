import pytest

from llm_cost.estimate import estimate_cost
from llm_cost.pricing import ModelPrice


def test_estimate_cost_prices_each_token_class():
    price = ModelPrice("m", input=5.0, output=25.0, cached_input=0.5, cache_write=6.25)
    breakdown = estimate_cost(
        price,
        input_tokens=12000,
        output_tokens=800,
        cached_input_tokens=52000,
        cache_write_tokens=9000,
    )
    assert breakdown.input_cost == pytest.approx(0.06)
    assert breakdown.output_cost == pytest.approx(0.02)
    assert breakdown.cached_input_cost == pytest.approx(0.026)
    assert breakdown.cache_write_cost == pytest.approx(0.05625)
    assert breakdown.total_cost == pytest.approx(0.06 + 0.02 + 0.026 + 0.05625)
    assert breakdown.total_tokens == 12000 + 800 + 52000 + 9000


def test_estimate_cost_scales_with_calls():
    price = ModelPrice("m", input=1.0, output=1.0)
    one = estimate_cost(price, input_tokens=1000, output_tokens=1000, calls=1)
    many = estimate_cost(price, input_tokens=1000, output_tokens=1000, calls=50)
    assert many.total_cost == pytest.approx(one.total_cost * 50)
    assert many.input_tokens == 50000
    assert many.calls == 50


def test_cost_per_call_is_zero_for_zero_calls():
    price = ModelPrice("m", input=1.0, output=1.0)
    breakdown = estimate_cost(price, calls=0)
    assert breakdown.cost_per_call == 0.0


def test_estimate_cost_rejects_negative_tokens():
    price = ModelPrice("m", input=1.0, output=1.0)
    with pytest.raises(ValueError):
        estimate_cost(price, input_tokens=-1)


def test_estimate_cost_rejects_non_integer_tokens():
    price = ModelPrice("m", input=1.0, output=1.0)
    with pytest.raises(ValueError):
        estimate_cost(price, input_tokens=1.5)


def test_to_dict_rounds_to_six_places():
    price = ModelPrice("m", input=3.0, output=15.0)
    breakdown = estimate_cost(price, input_tokens=1, output_tokens=1)
    data = breakdown.to_dict()
    assert data["input_cost"] == round(3.0 / 1000000.0, 6)
    assert data["total_cost"] == round(breakdown.total_cost, 6)
