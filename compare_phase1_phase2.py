import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument("--phase1-res", required=True)
    parser.add_argument("--phase2-res", required=True)
    args = parser.parse_args()

    with open(args.phase1_res, "r") as f:
        phase1_res = json.load(f)
    
    with open(args.phase2_res, "r") as f:
        phase2_res = json.load(f)

    phase2_avg = 0
    phase2_res_by_model = {}
    for v in phase2_res.values():
        phase2_avg += sum([succ for succ in v.values()])
        phase2_res_by_model.update(v)

    phase2_avg /= len(phase1_res)
    phase1_avg = sum([v for k, v in phase1_res.items() if k in phase2_res_by_model]) / len(phase2_res_by_model)
    print(f"Phase 1 average success rate: {phase1_avg:.4f}")
    print(f"Phase 2 average success rate: {phase2_avg:.4f}")

    grouping = {}
    for spe_id, v in phase2_res.items():
        grouping.update({model_id: spe_id for model_id in v.keys()})
    diff = [{"Model ID": k, "Success rate difference": phase2_res_by_model[k] - phase1_res[k], "gen": phase1_res[k], "spe": phase2_res_by_model[k], "spe_id": grouping[k]}
        for k in sorted(phase2_res_by_model.keys())]
    diff = pd.DataFrame(diff)
    barplot = seaborn.barplot(diff, y="Model ID", x="Success rate difference", color="lightblue")
    barplot.set(yticklabels=[])
    barplot.set(ylabel=None)
    barplot.set(xlabel=None)
    barplot.tick_params(left=False)
    plt.xlabel("Success rate difference", fontsize=14)
    plt.savefig("diff_dist.png")
    plt.cla()
    diff.to_csv("diff.csv", index=False)

    avg_std = 0

    for i in range(9):
        cur_spe_res = diff.loc[diff["spe_id"] == f"spe{i}"]
        cur_spe_res.sort_values("Success rate difference", inplace=True, ignore_index=True)

        avg_diff = cur_spe_res["Success rate difference"].mean()
        std_diff = cur_spe_res["Success rate difference"].std()
        avg_std += std_diff

        barplot = seaborn.barplot(cur_spe_res, x="Model ID", y="Success rate difference", color="lightblue")    
        barplot.set(ylabel=None)
        # barplot.set(xlabel=f"Specialist {i}", labelsize=14)
        barplot.tick_params(left=False)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel(f"Specialist {i}: {avg_diff:.4f}/{std_diff:.4f}", fontsize=14)
        plt.savefig(f"spe{i}_diff_dist.png")
        plt.cla()

    print(avg_std / 9)

    # spe_0_res = diff.loc[diff["spe_id"] == "spe0"]
    # spe_0_res.sort_values("Success rate difference", inplace=True, ignore_index=True)
    # barplot = seaborn.barplot(spe_0_res, x="Model ID", y="Success rate difference", color="lightblue")
    # # barplot.set(yticklabels=[])
    # barplot.set(ylabel=None)
    # barplot.set(xlabel=None)
    # barplot.tick_params(left=False)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.savefig("spe0_diff_dist.png")
    # plt.cla()

    # spe_5_res = diff.loc[diff["spe_id"] == "spe5"]
    # spe_5_res.sort_values("Success rate difference", inplace=True, ignore_index=True)
    # barplot = seaborn.barplot(spe_5_res, x="Model ID", y="Success rate difference", color="lightblue")
    # # barplot.set(yticklabels=[])
    # barplot.set(ylabel=None)
    # barplot.set(xlabel=None)
    # barplot.tick_params(left=False)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.savefig("spe5_diff_dist.png")
    # plt.cla()

if __name__ == "__main__":
    main()
