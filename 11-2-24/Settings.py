SimulationVariables = {
    "SimAgents" : 10,
    "AgentData" : [],
    "SafetyRadius" : 3,  # 2
    "VelocityInit" : 1.0,
    "AccelerationInit" : 0.0,
    "NeighborhoodRadius" : 7, # 10
    "VelocityUpperLimit" : 2.5,
    "AccelerationUpperLimit" : 5.0,
    "X" : 10,
    "Y" : 10,
    "dt" : 0.1,
    "EvalTimeSteps" : 3000,
    "LearningTimeSteps" : 20000000,
    "Episodes" : 10,
    "LearningRate": 0.003,
    "NumEnvs": 6,
    "MAX_COUNTER_VALUE": 2048
}

Files = {
    "Flocking" : 'Results\Flocking',
    "VelocityBased" : 'Results\Velocitybased\Result_',
    "Reynolds" : 'Results\ReynoldRl\Result_',
    "Logs" : "Results\Flocking\Testing\Rewards\Components"

}

RLVariables = {
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
    "Sim" : "Simulation",
    "InitPositions" : "CorrectConfigs\Config_",
    "Positions" : "Results\Flocking\Testing\Positions\Episode_3", 
    "Rewards" : "RewardTesting_Episode1",
    "EpRewards" : "Results",
    "Directory" : "CorrectConfigs",       #Config Generator
}


DDPGParam = {
    "Episodes" : 100,
    "gamma" : 0.99,
    "tau" : 0.001,
    "buffer_size" : 100000,
    "batch_size" : 64,
    "exploration_noise" : 0.1
}

Animation = {
    "OutputDirectory" : "",
    "InputDirectory" : "",
    "OutputFile" : "",
    "InputFile" : "",
    "TimeSteps" : 3000
}
