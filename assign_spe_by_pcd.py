import json
import os
import random
import sys
from argparse import ArgumentParser
sys.path.append("Pointnet_Pointnet2_pytorch/models")

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from pointnet2_cls_ssg import get_model


@torch.no_grad()
def extract_pcd_feats(points, model):
    feats = model(points, return_feats=True)
    feats = F.normalize(feats, dim=1)
    feats = feats.cpu().numpy().squeeze()

    return feats


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--succ-thresh", default=0.2, type=float,
        help="The success rate upper bound requiring specialist training.")
    parser.add_argument("--n-spe", default=3, type=int)
    parser.add_argument("--n-steps-spe", default=3000000, type=int,
        help="The number of training steps for each specialist.")
    args = parser.parse_args()
    args.seed = 42 * args.seed
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen.yaml")

    random.seed(42)
    np.random.seed(42)

    with open(f"logs/{args.env}/seed={args.seed}/eval_phase1_gen.json", "r") as f:
        eval_res = json.load(f)

    low_perf_models = []
    high_perf_models = []
    low_perf_succ = 0
    high_perf_succ = 0
    overall_perf_succ = 0
    for model_id, succ_rate in eval_res.items():
        overall_perf_succ += succ_rate
        if succ_rate < args.succ_thresh:
            low_perf_models.append(model_id)
            low_perf_succ += succ_rate
        else:
            high_perf_models.append(model_id)
            high_perf_succ += succ_rate

    model = get_model(40, normal_channel=False).cuda()
    weights = torch.load("Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth")["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()

    progress = tqdm(total=len(low_perf_models))
    features = {}
    for i in range(len(low_perf_models)):
        progress.set_description_str(f"{low_perf_models[i]}")
        pcd_path = f"data/partnet_mobility/dataset/{low_perf_models[i]}/point_sample/pts-10000.txt"
        pcd = np.loadtxt(pcd_path)
        pcd = torch.from_numpy(pcd.T[None, :]).float().cuda()
        res = extract_pcd_feats(pcd, model)
        features[low_perf_models[i]] = res

        progress.update()

    progress.close()
    feat_mat = np.stack([features[k] for k in sorted(list(features.keys()))])

    cluster_model = KMeans(args.n_spe)
    clusters = np.array(cluster_model.fit_predict(feat_mat))

    tsne = TSNE()
    feat_2d = tsne.fit_transform(feat_mat)

    for cluster_idx in np.unique(clusters):
        cur_feat = feat_2d[np.argwhere(clusters == cluster_idx).squeeze()]
        plt.scatter(cur_feat[:, 0], cur_feat[:, 1], label=cluster_idx)

    plt.legend()
    plt.savefig(f"{args.env}-tsne-by-pcd.png")

    grouping = {int(spe_id): [low_perf_models[idx] for idx in np.argwhere(clusters == spe_id).squeeze()]
        for spe_id in np.unique(clusters)}

    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen/phase1_gen.yaml")
    trial_name = cfg.trial_name.replace("phase1_gen", "phase2")
    for spe_id, model_ids in grouping.items():
        cfg.env.model_ids = ",".join(model_ids)
        cfg.trial_name = f"{trial_name}_{spe_id}"
        cfg.train.total_steps = args.n_steps_spe
        cfg.env.n_env_procs = 16
        with open(f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_id}_by_pcd.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)

    with open(f"logs/{args.env}/seed={args.seed}/spe_models_by_pcd.json", "w") as f:
        json.dump(grouping, f)


if __name__ == "__main__":
    main()
