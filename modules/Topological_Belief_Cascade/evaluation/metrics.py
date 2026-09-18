def average_event_ms(values: list[float]) -> float:
    return sum(values) / max(1, len(values))
