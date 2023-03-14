import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf

from evaluate import single_evaluate

# The generalist scores >=0.7 success rate on 20/60 models: ['5001', '5002', '5004', '5006', '5015', '5023', '5028', '5029', '5034', '5037', '5055', '5058', '5060', '5063', '5064', '5067', '5068', '5069', '5073', '5076'].
# Average success rates on these models: 0.8200.

# The generalist scores <0.7 success rate on 40/60 models: ['5000', '5005', '5007', '5010', '5011', '5012', '5014', '5016', '5018', '5020', '5021', '5024', '5025', '5027', '5030', '5033', '5035', '5038', '5039', '5040', '5041', '5043', '5044', '5045', '5046', '5047', '5048', '5049', '5050', '5051', '5052', '5053', '5056', '5057', '5061', '5062', '5065', '5070', '5072', '5075'].
# Average success rates on these models: 0.2212.

# Overall average success rates: 0.4208.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Config name.")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    res = {}

    for model_id in cfg.env.model_ids:
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, args.model_path)
        res[model_id] = succ_rate
    
    save_dir = os.path.dirname(args.model_path)
    with open(os.path.join(save_dir, "eval_gen_phase1_res.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
