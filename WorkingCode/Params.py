SimulationVariables = {
    "SimAgents": 20,
    "AgentData": [],
    "SafetyRadius": 5, 
    "NeighborhoodRadius": 10,
    "VelocityUpperLimit": 5.0,
    "VelocityInit": 1.0,
    "AccelerationUpperLimit": 1.0,
    "AccelerationInit": 0.0,
    "dt": 0.1,
    "LearningTimeSteps":600000,
    "EvalTimeSteps": 3000,
    "X": 10,
    "Y": 10,
    "NumEnvs": 6
}

RLVariables = {
    "Episodes": 10,
    "LearningRate": "",
    "Policies": {
                "MLP": ["MlpPolicy", "CnnPolicy", "MultiInputPolicy", "MultiInputCNNPolicy"],
                "LSTM": "LstmPolicy",
                "Attention": "AttentionPolicy",
                "ActorCritic": ["ActorCriticPolicy", "ActorCriticCnnPolicy", "ActorCriticLstmPolicy"],
                "NatureCNN_DQN": "NatureCnnPolicy",
                "SAC": "SACPolicy",
                "TD3": "TD3Policy",
                }
}

Results = {
    "Sim": "Simulation",
    "InitPositions": "Simulations\Config_0",
    "Positions": "agent_positionsTestAllignment_5.json",
    "Rewards": "RewardsTraining.json",
    "EpRewards": "EpisodeTraining"
}

Animation = {
    "OutputDirectory": "",
    "InputDirectory": "",
    "OutputFile": "",
    "InputFile": "",
    "TimeSteps":3000
}
