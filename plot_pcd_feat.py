import json
import os
import random
import sys
from argparse import ArgumentParser
from itertools import chain
sys.path.append("Pointnet_Pointnet2_pytorch/models")

import numpy as np
import pandas as pd
import seaborn as sns
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
    args = parser.parse_args()
    args.seed = 42 * args.seed

    random.seed(42)
    np.random.seed(42)

    with open(f"logs/{args.env}/seed={args.seed}/spe_models_by_pcd.json", "r") as f:
        group_by_pcd = json.load(f)
    with open(f"logs/{args.env}/seed={args.seed}/spe_models_by_orc.json", "r") as f:
        group_by_orc = json.load(f)

    model = get_model(40, normal_channel=False).cuda()
    weights = torch.load("Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth")["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()

    model_ids = list(chain(*[v for v in group_by_pcd.values()]))
    progress = tqdm(total=len(model_ids))
    features = {}
    for i in range(len(model_ids)):
        progress.set_description_str(f"{model_ids[i]}")
        pcd_path = f"data/partnet_mobility/dataset/{model_ids[i]}/point_sample/pts-10000.txt"
        pcd = np.loadtxt(pcd_path)
        pcd = torch.from_numpy(pcd.T[None, :]).float().cuda()
        res = extract_pcd_feats(pcd, model)
        features[model_ids[i]] = res

        progress.update()

    progress.close()
    feat_mat = np.stack([features[k] for k in sorted(list(features.keys()))])

    tsne = TSNE(perplexity=10)
    feat_2d = tsne.fit_transform(feat_mat)

    sns.set_theme()

    pcd_data = []
    for cluster_idx in group_by_pcd.keys():
        indices = [i for i, model_id in enumerate(model_ids) if model_id in group_by_pcd[cluster_idx]]
        cur_feat = feat_2d[indices]
        cur_feat = pd.DataFrame({"x": cur_feat[:, 0], "y": cur_feat[:, 1], "label": [f"Specialist {cluster_idx[-1]}"] * cur_feat.shape[0]})
        pcd_data.append(cur_feat)
    pcd_data = pd.concat(pcd_data, ignore_index=True)
    sns.scatterplot(pcd_data, x="x", y="y", hue="label")
    # plt.legend()
    plt.savefig(f"{args.env}-{args.seed}-tsne-by-pcd.png", dpi=600)
    plt.close()

    axes = ["x-axis", "y-axis", "z-axis"]
    orc_data = []
    for cluster_idx in group_by_orc.keys():
        indices = [i for i, model_id in enumerate(model_ids) if model_id in group_by_orc[cluster_idx]]
        cur_feat = feat_2d[indices]
        cur_feat = pd.DataFrame({"x": cur_feat[:, 0], "y": cur_feat[:, 1], "label": [axes[int(cluster_idx[-1])]] * cur_feat.shape[0]})
        orc_data.append(cur_feat)
    orc_data = pd.concat(orc_data, ignore_index=True)
    sns.scatterplot(orc_data, x="x", y="y", hue="label")
    # plt.legend()
    plt.savefig(f"{args.env}-{args.seed}-tsne-by-orc.png", dpi=600)


if __name__ == "__main__":
    main()
