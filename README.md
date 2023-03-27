# Explore Task Partitioning in State-Based Generalist-Specialist Learning
This is the final project of UCSD CSE 291A Winter 2023, authored by Kaiming Kuang, Jianyu Wang and Xiu Yuan.
# Usage
## WandB setup
Write a `wandb_cfg.yaml` file with your WandB configs in the root directory:
```yaml
key: <your_wandb_key>
entity: <your_account_name>
project: <your_project_name>
```
## Phase I: Generalist training
Train the generalist agent using SAC algorithm on the TurnFaucet task:
```bash
python main.py --env=TurnFaucet-v2 --seed=1
```
Find where the generalist training plateaus:
```bash
python find_plateau.py --env=TurnFaucet-v2 --seed=1
```
`find_plateau.py` gives you the weight path where the generalist training plateaus. Now evaluate this weight on 60 different models of TurnFaucet:
```bash
python evaluate_gen_phase1.py --env=TurnFaucet-v2 --seed=1
```
The evaluation outputs a `eval_gen_phase1_res.json` with success rates on each of the 60 faucet models. Now find out which models should be refined and assign specialist:
```bash
python random_assign_spe.py --env=TurnFaucet-v2 --seed=1
``` 

## Phase II: Specialist training and demo collection
For example, you can train the first specialist with the following command:
```bash
python main.py --env=TurnFaucet-v2 --seed=1 --spe-idx=0
```
You may spawn multiple specialist training jobs at the same time. Evaluate all specialists after training:
```bash
python evaluate_spe_phase2.py --env=TurnFaucet-v2 --seed=1
```
Now collect demos fron the generalist and specialists:
```bash
python collect_demos.py --env=TurnFaucet-v2 --seed=1
```

## Phase III: Generalist fine-tuning
Train the generalist on the demonstrations collected:
```bash
python train_phase3_gen.py --env=TurnFaucet-v2 --seed=1
```
Evaluate the fine-tuned generalist:
```bash
python evaluate_phase3_gen.py --env=TurnFaucet-v2 --seed=1
```