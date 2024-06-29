   def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)
                
                 if distance <= SimulationVariables["SafetyRadius"]:
                    CohesionReward -= 10  

                 elif SimulationVariables["SafetyRadius"] < distance < midpoint:
                     ratio = (midpoint - SimulationVariables["SafetyRadius"])
                     CohesionReward += (10 / ratio) * (distance - SimulationVariables["SafetyRadius"])

                 elif midpoint <= distance < SimulationVariables["NeighborhoodRadius"]:
                     ratio = (SimulationVariables["NeighborhoodRadius"] - midpoint)
            #        CohesionReward += (10 / ratio) * (SimulationVariables["NeighborhoodRadius"] - distance)
