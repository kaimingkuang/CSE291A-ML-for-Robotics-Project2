# TODO
- Analyze Phase II results. Some tasks still have 0% success rates.
- Extract features of faucets using https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth.
- Specialist evaluation and demo collection.
- Run DAPG on generalist with specialist demos.
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
python main.py --cfg=turnfaucet_gen_phase1_sac
```
Find where the generalist training plateaus:
```bash
python find_plateau.py --tb-path=<tensorboard_event_path>
```
`find_plateau.py` gives you the weight path where the generalist training plateaus. Now evaluate this weight on 60 different models of TurnFaucet:
```bash
python evaluate_gen_phase1.py --cfg=turnfaucet_gen_phase1_sac --model-path=<plateau_weight_path>
```
The evaluation outputs a `eval_gen_phase1_res.json` with success rates on each of the 60 faucet models. Now find out which models should be refined and assign specialist:
```bash
python assign_spe.py --eval-json=<your_log_dir/eval_gen_phase1_res.json> --cfg=turnfaucet_gen_phase1_sac --spe-init=<plateau_weight_path>
``` 
It shows the subset of models that the generalist agent performs well/badly (the default threshold is success rate >= 0.7). Then the low-performance group is randomly divided into several subsets, each of which is assigned to one specialist. For each specialist, a training config yaml file is generated for later specialist training. For example, 46 models are assigned to 9 specialists. Corresponding training .yaml files are named from `turnfaucet_spe_phase2_sac_spe0.yaml` to `turnfaucet_spe_phase2_sac_spe8.yaml`.

## Phase II: Specialist training and demo collection
For example, you can train the first specialist with the following command:
```bash
python main.py --cfg=turnfaucet_gen_phase1_sac_spe0
```