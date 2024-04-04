# Alignment


if (len(neighbor_velocities) > 0):
  
    average_velocity = np.mean(neighbor_velocities, axis=0) 
    desired_orientation = average_velocity - agent.velocity   
    orientation_diff = np.arctan2(
    desired_orientation[1], desired_orientation [0]) 
    - np.arctan2(agent.velocity[1], agent.velocity[0])

    if orientation_diff > np.pi:
        orientation_diff -= 2 * np.pi
    elif orientation_diff < 0:
          orientation_diff += 2 * np.pi

    total_reward = 10*(1 - np.abs(orientation_diff))


# Separation
if neighbor_positions:  # Check if there are neighbors
    for neighbor_position in neighbor_positions:
      
    distance = np.linalg.norm(agent.position - neighbor_position)
    if distance < SimulationVariables["SafetyRadius"]:
        total_reward -= 10
    elif SimulationVariables["SafetyRadius"] < distance < SimulationVariables["NeighborhoodRadius"]:
          alpha = 0.99  # Adjust this parameter as needed
          total_reward += np.exp(-alpha * distance)

