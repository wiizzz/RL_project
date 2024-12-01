# RL_project
NTU RL2023 final project

## Usage
### Training manager (2 actions)
1. In code/manager/ , modify the self.prompt_gamma in manager_env.py. 
2. Modify the import env file on the top of  manager_train.py 
3. Run the manager_train.py
### Training manager (4 actions)
1. In code/manager/ , modify the self.prompt_gamma in manager_env_4actions.py. 
2. Modify the import env file on the top of  manager_train.py
3. Run the manager_train.py
### Evaluating manager
In code/manager/, modify the import env file on the top of manager_eval.py, and change the map you want in the 'create_env()' function in manager_env.py, and run manager_eval.py

### Training pretrain skill
In code/pretrain_skill, like training procedure above, modify *env.py,new_train.py in order 
(navigation's env file is new_env.py) 
### Evaluating pretrain skill
In code/pretrain_skill/, modify the import env file on the top of eval.py ,and run eval.py

### Generating llm plan
In code/llm_prompt/, run the first_prompt.py and second_prompt.py