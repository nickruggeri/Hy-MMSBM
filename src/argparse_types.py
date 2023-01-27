from __future__ import annotations


def int_or_none(n: str) -> int | None:
    """Convert argparse argument to either integer or None."""
    if n.lower() == "none":
        return None
    return int(n)


def float_or_none(x: str) -> float | None:
    """Convert argparse argument to either float or None."""
    if x.lower() == "none":
        return None
    return float(x)


def float_or_str(x: str) -> float | str | None:
    try:
        return float(x)
    except ValueError:
        return x


def bool_type(x: str) -> bool:
    if x.lower() == "false" or x == "0":
        return False
    elif x.lower() == "true" or x == "1":
        return True
    raise ValueError("Value cannot be converted to boolean:", x)
