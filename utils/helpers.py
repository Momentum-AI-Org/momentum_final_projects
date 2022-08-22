def round_with_prec(x: float, prec: int = 2) -> float:
    """Return a number rounded to the correct precision"""
    dec_prec_value = 10**prec
    return round(x * dec_prec_value) / dec_prec_value
