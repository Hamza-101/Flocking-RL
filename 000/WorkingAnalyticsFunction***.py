def generateAnalytics():
    # Loop through episodes
    for episode in range(SimulationVariables["Episodes"]):
        # Initialize dictionaries to store velocities and accelerations for this episode
        velocities_dict = {}
        accelerations_dict = {}

        # Read velocities and accelerations from JSON files
        with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'r') as f:
            episode_velocities = json.load(f)
        with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'r') as f:
            episode_accelerations = json.load(f)

        # Append velocities and accelerations to dictionaries
        for agent_id in range(len(env.agents)):
            velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))
            accelerations_dict.setdefault(agent_id, []).extend(episode_accelerations.get(str(agent_id), []))

        # Plot velocities
        plt.figure(figsize=(10, 5))
        plt.clf()  # Clear the current figure
        for agent_id in range(len(env.agents)):
            agent_velocities = np.array(velocities_dict[agent_id])
            smoothed_velocities = savgol_filter(agent_velocities, window_length=21, polyorder=3, axis=0)
            velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)  # Magnitude of velocities
            plt.plot(velocities_magnitude, label=f"Agent {agent_id+1}")
        plt.title(f"Smoothed Velocity - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedVelocity.png")

        # Plot accelerations
        plt.figure(figsize=(10, 5))
        plt.clf()  # Clear the current figure
        for agent_id in range(len(env.agents)):
            agent_accelerations = np.array(accelerations_dict[agent_id])
            smoothed_accelerations = savgol_filter(agent_accelerations, window_length=21, polyorder=3, axis=0)
            accelerations_magnitude = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)  # Magnitude of accelerations
            plt.plot(accelerations_magnitude, label=f"Agent {agent_id+1}")
        plt.title(f"Smoothed Acceleration - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedAcceleration.png")
