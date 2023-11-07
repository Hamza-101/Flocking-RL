SimulationVariables = {
    "SimAgents" : 20,
    "AgentData" : [],
    "SafetyRadius" : 2, 
    "NeighborhoodRadius" : 10,
    "VelocityUpperLimit" : 5.0,
    "VelocityInit" : 1.0,
    "AccelerationUpperLimit" : 1.0,
    "AccelerationInit": 0.0,
    "dt" : 0.1,
    "TimeSteps" : 1000,
    "X" : 10,
    "Y" : 10,
    "Reward" : "", # Check this
    "TrainFile" : "Simulations\Config_0",
    "TestFile" : "Simulations\Test"
}

Results = {
    "Directory" : "Simulations",
    "Sim" : "Simulation", # Change this
    "InitPositions" : "Simulations\Config_2",
    "FinalPositions" : "", # Change this
    "SimDetails" : SimulationVariables,
    "Algorithm" : "", # Change this
    "TotalReward": [] # Sum Array when needed
}

