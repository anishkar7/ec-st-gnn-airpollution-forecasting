def aqi_category(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def aqi_color(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "#2c7a3f"
    if aqi_value <= 100:
        return "#8bc34a"
    if aqi_value <= 200:
        return "#f6b100"
    if aqi_value <= 300:
        return "#f17c00"
    if aqi_value <= 400:
        return "#df3f2f"
    return "#7e0023"


def safe_pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100.0
