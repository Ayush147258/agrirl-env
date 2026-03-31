def act(observation):
    """
    Baseline policy for AgriRL environment.
    """

    soil = observation["soil_moisture"]
    growth = observation["crop_growth"]

    # Harvest when growth is high
    if growth > 60:
        return {"action": "harvest"}

    # Maintain soil moisture
    if soil < 40:
        return {"action": "irrigate"}

    # Boost growth early
    if growth < 30:
        return {"action": "fertilize"}

    return {"action": "wait"}

def explain(obs):
    if obs["crop_growth"] > 60:
        return "Harvest because crop growth is high → maximize reward"
    if obs["soil_moisture"] < 40:
        return "Irrigate because soil moisture is low"
    if obs["crop_growth"] < 30:
        return "Fertilize to boost early growth"
    return "Wait to maintain balance"