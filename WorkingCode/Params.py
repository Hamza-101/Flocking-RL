SimulationVariables = {
    "SimAgents": 10,
    "AgentData": [],
    "SafetyRadius": 3, 
    "VelocityInit": 1.0,
    "AccelerationInit": 0.0,
    "NeighborhoodRadius": 7,
    "VelocityUpperLimit": 5.0,
    "AccelerationUpperLimit": 1.0,
    "X": 10,
    "Y": 10,
    "dt": 0.1,
    "EvalTimeSteps": 3000,
    "LearningTimeSteps":1000000,
    "TotalEps" : 10,
    "NumEnvs": 6,
    "MAX_COUNTER_VALUE": 2048
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
    "InitPositions": "CorrectConfigs\Config_",
    "Positions": "FlockingCustom1.json", 
    "Rewards": "RewardTesting_Episode",
    "EpRewards": "EpisodeTraining",
    "Directory": "CorrectConfigs"       #Config Generator
}


DDPGParam={
    "Episodes" : 100,
    "gamma" : 0.99,
    "tau" : 0.001,
    "buffer_size" : 100000,
    "batch_size" : 64,
    "exploration_noise" : 0.1
}

Animation = {
    "OutputDirectory": "",
    "InputDirectory": "",
    "OutputFile": "",
    "InputFile": "",
    "TimeSteps":3000
}