import pandas as pd


def source_attribution(latest_row: pd.Series) -> pd.DataFrame:
    # Deterministic placeholder split for demo stability.
    traffic = min(55.0, max(25.0, 25.0 + float(latest_row.get("no2", 80.0)) / 6.0))
    weather = min(45.0, max(20.0, 35.0 + (50.0 - float(latest_row.get("humidity", 50.0))) / 5.0))
    regional = max(10.0, 100.0 - traffic - weather)

    return pd.DataFrame(
        {
            "Source": ["Traffic", "Meteorology", "Regional Transport"],
            "Contribution (%)": [round(traffic, 1), round(weather, 1), round(regional, 1)],
        }
    )


def causal_graph_dot() -> str:
    return """
    digraph {
        rankdir=LR;
        Traffic -> PM25 [label=" + "];
        WindSpeed -> PM25 [label=" - "];
        Humidity -> PM25 [label=" + "];
        Temperature -> Ozone [label=" + "];
        Ozone -> PM25 [label=" + "];
    }
    """


def run_counterfactual(latest_pm25: float, traffic_cut: int, industry_cut: int) -> tuple[float, float]:
    projected_drop = (traffic_cut * 0.35) + (industry_cut * 0.25)
    projected_pm25 = max(0.0, latest_pm25 * (1 - projected_drop / 100))
    return projected_pm25, projected_drop


def policy_recommendation(latest_pm25: float) -> tuple[str, str]:
    if latest_pm25 >= 300:
        return (
            "Activate graded response: traffic rationing + construction curbs + emergency advisories",
            "High",
        )
    if latest_pm25 >= 200:
        return (
            "Target heavy-duty vehicle restrictions and dust suppression in critical corridors",
            "Medium",
        )
    return ("Maintain monitoring mode with preventive control actions", "Medium")
