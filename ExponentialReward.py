import numpy as np

def compute_cohesion_reward(agent_position, neighbor_positions, SimulationVariables):
    CohesionReward = 0
    midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

    for neighbor_position in neighbor_positions:
        distance = np.linalg.norm(agent_position - neighbor_position)
        
        if distance <= SimulationVariables["SafetyRadius"]:
            CohesionReward -= 10
        
        elif SimulationVariables["SafetyRadius"] < distance < midpoint:
            ratio = (distance - SimulationVariables["SafetyRadius"]) / (midpoint - SimulationVariables["SafetyRadius"])
            CohesionReward += 10 * np.exp(ratio - 1)
        
        elif midpoint <= distance < SimulationVariables["NeighborhoodRadius"]:
            ratio = (distance - midpoint) / (SimulationVariables["NeighborhoodRadius"] - midpoint)
            CohesionReward += 10 * np.exp(-ratio)

    return CohesionReward
