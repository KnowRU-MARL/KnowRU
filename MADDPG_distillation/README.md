# The code is an implementation of KnowRU based on MADDPG, you should firstly install the MADDPG(https://github.com/DKuan/MADDPG_torch). 
# Before you begin to reuse knowledge, you should train some policy models firstly.

## Installation
known dependencies: Python(3.6.8), OpenAI Gym(0.10.5), Pytorch(1.1.0), Numpy(1.17.3)    
Install the MPE(Multi-Agent Particle Environments) as the readme of OpenAI (or the blog of mine).    

# Structure
./main_openai_KD.py: Main func

./arguments.py: Init the par for game, training and saving.

./distillation.py: Some functions for distillation.

./model.py: Init the model for the agent.

./replay_buffer.py: Save the memory for all the agents.

./enjoy_split.py: A templete for testing the model trained in the 'main_openai.py'.

./base_model: Save the previous policy models for knowledge reusing.
