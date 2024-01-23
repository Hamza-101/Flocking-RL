def CollisionPenalty(self, agent, neighbor_indices):
        Collision = False
        CollisionPenalty = 0
        distances=[]

        neighbor_agents = [self.agents[idx] for idx in neighbor_indices]

        for n_agent in neighbor_agents:  
            distance = np.linalg.norm(agent.position - n_agent.position)
            distances.append(distance)

        for distance in distances:
            if distance < SimulationVariables["SafetyRadius"]:            
                CollisionPenalty -= 800
                Collision = True

        return CollisionPenalty, Collision
    
    def calculate_CohesionReward(self, agent, neighbor_indices):
        CohesionReward = 0
        distances = []
        Collision = False

        # Find why working
        for n_position in neighbor_indices:
            distance = np.linalg.norm(agent.position - n_position)
            distances.append(distance)

        for distance in distances:                
            if  distance <= SimulationVariables["NeighborhoodRadius"]:
                CohesionReward += 500 
            elif (distance > SimulationVariables["NeighborhoodRadius"]):
                CohesionReward += -(500 * (distance - SimulationVariables["NeighborhoodRadius"]))  # Incrementally decrease rewards
           
        return CohesionReward
    
    def calculate_AlignmentReward(self, agent, neighbor_velocities):
        reward_scale = 200  # Adjust this scale as needed

        desired_direction = self.alignment_calculation(agent, neighbor_velocities)
        orientation_diff = np.arctan2(desired_direction[1], desired_direction[0]) - np.arctan2(agent.velocity[1], agent.velocity[0])

        # Normalize orientation_diff between [0, π]
        orientation_diff = (orientation_diff + np.pi) % (2 * np.pi)

        if orientation_diff > np.pi:
            orientation_diff = 2 * np.pi - orientation_diff

        AlignmentReward = reward_scale * (1 - (orientation_diff / np.pi))

        return AlignmentReward
  