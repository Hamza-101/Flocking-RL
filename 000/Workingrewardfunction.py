 def alignment_reward(self, agent, neighbor_velocities, neighbor_positions):
        multiplier=1
        total_reward=0
        outofflock=False

        if(len(neighbor_positions) > 0):
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)
                    
                if distance < SimulationVariables["SafetyRadius"]:
                    # Large penalty to discourage agents from getting too close
                    total_reward -= 10 
                    multiplier = 1
                    
                elif SimulationVariables["SafetyRadius"] < distance < SimulationVariables["NeighborhoodRadius"]:
                    multiplier=10
                    # Decaying exponential function for rewards
                    alpha = 0.1  # Adjust this parameter as needed
                    total_reward += np.exp(-alpha * distance)

            
            if (len(neighbor_velocities) > 0):
                average_velocity = np.mean(neighbor_velocities, axis=0)
                desired_orientation = average_velocity - agent.velocity   
                orientation_diff = np.arctan2(desired_orientation[1], desired_orientation[0]) - np.arctan2(agent.velocity[1], agent.velocity[0])

                if orientation_diff > np.pi:
                    orientation_diff -= 2 * np.pi
                elif orientation_diff < 0:
                    orientation_diff += 2 * np.pi

                alignment = 1 - np.abs(orientation_diff)

                if (alignment < 0.5):
                    total_reward -= 50 * (alignment)
                else:
                    total_reward += 50 * (alignment)

        else:
            # If no neighbors, encourage agent to find flock
            total_reward -=50
            outofflock=True

        return total_reward, outofflock
  
