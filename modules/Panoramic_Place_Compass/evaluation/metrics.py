def circular_error_deg(prediction: float, target: float) -> float:
    return abs((float(prediction) - float(target) + 180.0) % 360.0 - 180.0)
