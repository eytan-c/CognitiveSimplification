import argparse
import math
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from typing import Any, List

import pathlib
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import seaborn as sn
import matplotlib.pyplot as plt
import tqdm
import scipy.stats

from util.article_plots import set_size, column_width, text_width

# TODO: Fix all file reference
# TODO: Import CSVs for analysis
# TODO: Add usage descriptions
# TODO: Clean comment outs

_ACTION_COLUMNS = ["PROX", "REPHRASE", "DEL", "ADD", "EXAMPLE", "EXPLAIN", "EXPLICIT", "REORDER", "SPLIT"]

_PARAPHRASE = ["simple_synonym", "word2phrase", "phrase2word", "phrase2phrase"]

_NUMERIC = ["ter", "token_length_ratio", "nbchars_ratio", "levsim", "wordrank_ratio", "deptree_depth_ratio"]

_OTHER = ["percent_deleted_unused", "percent_added_unused", "reg_nbchars", "sim_nbchars", "reg_wordrank",
          "sim_wordrank", "deptree_depth_ratio"]

_DS_GROUPS = {"ASSET": ("asset-test-all", "asset-valid-all"),
              "Cognitive": ("dhcs", "disability_fest_manual", "uncrpd"),
              "Newsela-Manual": ("newsela-manual-dev-all", "newsela-manual-test-all", "newsela-manual-train-all"),
              "Wiki-Manual": ("wiki-manual-dev", "wiki-manual-test", "wiki-manual-train"),
              "Wiki-Auto": ("wiki_auto-train-combined", "wiki_auto-valid-combined")}

_DS_REMAP = {'asset-test-all': "ASSET test",
             'asset-valid-all': "ASSET valid",
             'dhcs': "DHCS",
             'uncrpd': "UNCRPD",
             'disability_fest_manual': "FestAblitity",
             'newsela-manual-dev-all': "NewselaManual dev",
             'newsela-manual-test-all': "NewselaManual test",
             'newsela-manual-train-all': "NewselaManual train",
             'wiki-manual-dev': "WikiManual dev",
             'wiki-manual-test': "WikiManual test",
             'wiki-manual-train': "WikiManual train",
             'wiki_auto-train-combined': "WikiAuto train",
             'wiki_auto-valid-combined': "WikiAuto valid"}
_DS_REMAP_SHORT = {'asset-test-all': "ASSET-ts",
                   'asset-valid-all': "ASSET-vl",
                   'dhcs': "DHCS",
                   'uncrpd': "UN-ts",
                   'disability_fest_manual': "FA-ts",
                   'newsela-manual-dev-all': "NewM-dv",
                   'newsela-manual-test-all': "NewM-ts",
                   'newsela-manual-train-all': "NewM-tr",
                   'wiki-manual-dev': "WikiM-dv",
                   'wiki-manual-test': "WikiM-ts",
                   'wiki-manual-train': "WikiM-tr",
                   'wiki_auto-train-combined': "WikiA-tr",
                   'wiki_auto-valid-combined': "WikiA-vl"}

_DS_ORDER = ["UNCRPD", "FestAblitity", "NewselaManual test", "NewselaManual dev", "NewselaManual train", "ASSET test",
             "ASSET valid", "WikiAuto valid", "WikiAuto train", "WikiManual test", "WikiManual dev", "WikiManual train"]
_DS_ORDER_SHORT = ["UN-ts", "FA-ts", "NewM-ts", "NewM-dv", "NewM-tr", "ASSET-ts",
                   "ASSET-vl", "WikiA-vl", "WikiA-tr","WikiM-ts",  "WikiM-dv", "WikiM-tr"]

_DS_COLOR_MAP = {'asset-test-all': "r",
                 'asset-valid-all': "r",
                 'disability_fest_manual': "g",
                 'uncrpd': "g",
                 'newsela-manual-dev-all': "b",
                 'newsela-manual-test-all': "b",
                 'newsela-manual-train-all': "b",
                 'wiki-manual-dev': "r",
                 'wiki-manual-test': "r",
                 'wiki-manual-train': "r",
                 'wiki_auto-train-combined': "r",
                 'wiki_auto-valid-combined': "r"}
# _CMAP = "vlag"
_CMAP = sn.color_palette("cividis", as_cmap=True)
_SCALER = MinMaxScaler()

# plt.style.use("fivethirtyeight")
#
plt.rcParams["font.size"] = 12


# plt.rcParams["figure.figsize"] = (10, 10)
# plt.rcParams['axes.facecolor'] = 'white'


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def mean_pointwise_jsd(p, q):
    assert len(q) == len(p)
    l = len(p)
    return sum_pointwise_jsd(p, q) / l


def sum_pointwise_jsd(p, q):
    assert len(q) == len(p)
    l = len(p)
    result = [0 for _ in range(l)]
    for i in range(l):
        result[i] = jensen_shannon_distance([p[i], 1 - p[i]], [q[i], 1 - q[i]])
    return sum(result)


def series_norm(s, t, order=2):
    assert len(s) == len(t)
    l = len(s)
    result = [0 for _ in range(l)]
    for i in range(l):
        result[i] = np.sqrt(np.power(s[i] - t[i], 2))
    return sum(result) / l


def show_action_histograms(datapath: pathlib.Path, analysis_type="full", title="Action_frequencies", save_fig=False):
    distributions = {}
    actions = {}
    if analysis_type == "drop":
        title = f"{title}_drop"
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            if analysis_type == "drop":
                distributions[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].mean()
                actions[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].sum(axis=1)
            else:
                distributions[ds_name] = df[_ACTION_COLUMNS].mean()
                actions[ds_name] = df[_ACTION_COLUMNS].sum(axis=1)
    dist_df = pd.DataFrame([{"dataset": _DS_REMAP[k], "action": o, "prob": e} for k, v in
                            sorted(distributions.items(), key=lambda pair: _DS_ORDER.index(_DS_REMAP[pair[0]])) for o, e
                            in zip(v.index, v)])
    act_df = pd.DataFrame([{"dataset": _DS_REMAP[k], "idx": o, "number": e} for k, v in
                            sorted(actions.items(), key=lambda pair: _DS_ORDER.index(_DS_REMAP[pair[0]])) for o, e
                            in zip(v.index, v)])

    fig = plt.figure(figsize=set_size(text_width, subplots=(7, 4)))
    grid = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1])
    for i in range(4):
        for j in range(3):
            if i != 0 or j != 0:
                fig.add_subplot(grid[i, j], sharex=fig.axes[0])
            else:
                fig.add_subplot(grid[i, j])

    axs = fig.axes
    col = 0
    for ds in [_DS_REMAP[k] for k in actions.keys()]:
        g = sn.histplot(act_df[act_df["dataset"] == ds]["number"],ax=axs[col],bins=7)
        g.set(xlabel=None, ylabel=None)
        if ds.startswith("Newsela"):
            g.set(title=f"NewselaM {ds.split(' ')[1]}")
        elif ds.startswith("WikiM"):
            g.set(title=f"WikiM {ds.split(' ')[1]}")
        else:
            g.set(title=ds)
        col+=1
    plt.suptitle("Number of actions per SI")
    if save_fig:
        plt.savefig(f"{title}_histograms", bbox_inches="tight")
    else:
        plt.show(bbox_inches="tight")

    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=set_size(text_width, subplots=(5, 3)))

    grid = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.2])
    for i in range(3):
        for j in range(3):
            fig.add_subplot(grid[i, j])
    fig.add_subplot(grid[3, :])
    axs = fig.axes
    row, col = 0, 0

    #  Color blind colors
    clrs = [sn.color_palette("Oranges",3)[1],
            sn.color_palette("Oranges", 3)[2],
            sn.color_palette("Blues",3)[0],
            sn.color_palette("Blues",3)[1],
            sn.color_palette("Blues",3)[2],
            sn.color_palette("Greens",8)[1],
            sn.color_palette("Greens",8)[2],
            sn.color_palette("Greens",8)[3],
            sn.color_palette("Greens",8)[4],
            sn.color_palette("Greens",8)[5],
            sn.color_palette("Greens",8)[6],
            sn.color_palette("Greens",8)[7],
            ]
    clrs = ['#999999', '#777777', '#BA8DB4', '#AA6F9E', '#994F88', '#1965B0', '#437DBF', '#6195CF', '#7BAFDE', '#4EB265', '#90C987', '#CAE0AB']

    for act in _ACTION_COLUMNS:
        g = sn.barplot(x="action", y="prob", hue="dataset", palette=clrs,
                       data=dist_df[dist_df["action"] == act], ax=axs[col])
        g.legend_.remove()
        g.set(xlabel=None, ylabel=None)
        g.tick_params(bottom=False)
        col += 1
        # if act == "REPHRASE":
        #     g.set_ylim(0.7, 1)

    h, l = axs[0].get_legend_handles_labels()
    axs[-1].legend(h, l, ncol=3, loc="upper center", fontsize="x-small")
    axs[-1].set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    axs[-1].tick_params(left=None, bottom=None)
    axs[-1].set_frame_on(False)
    # g = sn.catplot(x="dataset", y="prob", hue="dataset", col="action", data=dist_df)
    # plt.title(title)
    plt.suptitle("Simplification Operation Probabilities")
    if save_fig:
        plt.savefig(f"{title}_per_Actions", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')


def show_action_distribution_distances(datapath: pathlib.Path, distance_func=mean_pointwise_jsd, analysis_type="full",
                                       add_point_labels=False, annotate_hm=False, save_fig=False,
                                       title="JSD_action_distances"):
    distributions = {}
    if analysis_type == "drop":
        title = f"{title}_drop"
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            if analysis_type == "drop":
                distributions[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].mean()
            else:
                distributions[ds_name] = df[_ACTION_COLUMNS].mean()
    distances = {s: {d: distance_func(distributions[s], distributions[d]) for d in distributions.keys()} for s in
                 distributions.keys()}
    dist_df = pd.DataFrame(distances)
    pca_df = pd.DataFrame(distributions)
    pca = PCA(n_components=2)
    pca.fit(pca_df)
    # pca = TSNE(n_components=2, perplexity=5)
    # pca.fit(pca_df.T)
    # sn.heatmap(dist_df, cmap=_CMAP)
    plt.cla()
    plt.clf()
    # fig = plt.figure(figsize=set_size(column_width))
    pca_df = pd.DataFrame(pca.components_.T, index=pca_df.columns)
    # pca_df = pd.DataFrame(pca.embedding_, index=pca_df.columns)
    pca_df = pca_df - pca_df.loc["disability_fest_manual", :]
    pca_df["Corpus"] = ["ASSET", "ASSET", "FestAbility", "UNCRPD", "NewselaManual", "NewselaManual", "NewselaManual",
                        "WikiManual", "WikiManual", "WikiManual", "WikiAuto", "WikiAuto"]
    g = sn.scatterplot(x=pca_df[0], y=pca_df[1], hue=pca_df["Corpus"], s=300, style=pca_df["Corpus"])
    g.set_xlabel("")
    g.set_ylabel("")
    plt.ylim(-0.5, 1.1)
    plt.xlim(-0.2, 0.22)

    for lh in g.legend_.legendHandles:
        lh.set_sizes([100])
    # g.set_xticklabels([])
    # g.set_yticklabels([])
    # plt.grid()
    if analysis_type == "drop":
        if add_point_labels:
            offsets = [(0.008, 0.006), (-0.053, -0.03), (0, 0), (0.006, -0.05), (0.005, 0.005), (-0.05, -0.005),
                       (0.01, -0.025), (0.01, -0.025), (-0.051, -0.05), (-0.051, 0.004), (-0.053, -0.03)]
            for x, y, s, o in zip(pca_df[0], pca_df[1], dist_df["disability_fest_manual"], offsets):
                if s != 0:
                    g.text(x=x + o[0], y=y + o[1], s=f"{s:.3f}")
        plt.title(f"Fine Tuning Operation Probabilities PCA")
    else:
        if add_point_labels:
            offsets = [(-0.003, 0.06), (-0.005, -0.11), (0, 0), (0.005, 0.005), (0.005, 0.005), (0.005, 0.005),
                       (0.01, 0.005), (-0.01, -0.12), (-0.05, 0.005), (-0.05, 0.005), (-0.005, -0.11)]
            for x, y, s, o in zip(pca_df[0], pca_df[1], dist_df["disability_fest_manual"], offsets):
                if s != 0:
                    g.text(x=x + o[0], y=y + o[1], s=f"{s:.3f}")
        plt.title(f"Full Corpus Operation Probabilities PCA")
    if save_fig:
        plt.savefig(f"{title}_pca", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')

    dist_df = dist_df.rename(columns=_DS_REMAP_SHORT, index=_DS_REMAP_SHORT)
    dist_df = dist_df.reindex(columns=_DS_ORDER_SHORT, index=_DS_ORDER_SHORT)
    plt.cla()
    plt.clf()
    # f, ax = plt.subplots(figsize=set_size(column_width))
    # g = sn.heatmap(scale_df(dist_df), cmap=_CMAP)
    # fig = plt.figure(figsize=set_size(column_width))
    max_jsd = 0.341 if dist_df.max().max() < 0.341 else dist_df.max().max()
    annot_kw = {"fontsize": 7.0, "ma": "center"} if annotate_hm else {}
    g = sn.heatmap(dist_df, cmap=_CMAP,
                   # vmin=0, vmax=max_jsd,
                   annot=annotate_hm, annot_kws=annot_kw, fmt=".3f")
    # g.set_yticklabels(g.get_yticklabels(), rotation=0)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    # cbar = g.collections[0].colorbar
    # cbar.set_ticks([0, .2, .4, .6, .8, 1])
    # cbar.set_ticklabels(["min", "20%", "40%", "60%", "80%", "max"])
    # cb = plt.gcf().axes[-1]
    # cb.set
    # sn.clustermap(dist_df, row_cluster=False)
    # plt.title(title, fontsize=15)

    # print(f"{title}: max - {dist_df.max().max():.3f}, min - {dist_df.min().min():.3f}")
    plt.title(title)
    if save_fig:
        plt.savefig(f"{title}.png", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')


def show_dist_and_corr_together(datapath: pathlib.Path, analysis_type="full",
                                distance_func=mean_pointwise_jsd, save_fig=False):
    distributions = {}
    cors = {}
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            if analysis_type == "drop":
                distributions[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].mean()
            else:
                distributions[ds_name] = df[_ACTION_COLUMNS].mean()
            if analysis_type == "drop":
                cors[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].corr()
            else:
                cors[ds_name] = df[_ACTION_COLUMNS].corr()
    distances = {s: {d: distance_func(distributions[s], distributions[d]) for d in distributions.keys()} for s in
                 distributions.keys()}
    cor_distances = {s: {d: np.linalg.norm(cors[s] - cors[d]) for d in cors.keys()} for s in cors.keys()}

    dist_df = pd.DataFrame(distances)
    cor_dist_df = pd.DataFrame(cor_distances)

    dist_df = dist_df.rename(columns=_DS_REMAP_SHORT, index=_DS_REMAP_SHORT)
    dist_df = dist_df.reindex(columns=_DS_ORDER_SHORT, index=_DS_ORDER_SHORT)
    cor_dist_df = cor_dist_df.rename(columns=_DS_REMAP_SHORT, index=_DS_REMAP_SHORT)
    cor_dist_df = cor_dist_df.reindex(columns=_DS_ORDER_SHORT, index=_DS_ORDER_SHORT)

    plt.cla()
    plt.clf()

    fig = plt.figure(figsize=set_size(text_width))

    grid = GridSpec(1, 3, figure=fig, width_ratios=[50, 50, 1])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1], sharey=ax1)
    cax = fig.add_subplot(grid[0, 2])

    sn.heatmap(scale_df(dist_df), cmap=_CMAP, ax=ax1, cbar_ax=cax)
    sn.heatmap(scale_df(cor_dist_df), cmap=_CMAP, ax=ax2, cbar_ax=cax)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    ax2.yaxis.set_visible(False)
    ax1.set_title(f"{distance_func.__name__}", fontsize=8)
    ax2.set_title("correlation matrix $\ell_2$-distance", fontsize=8)
    plt.suptitle("Distances between datasets")

    if save_fig:
        plt.savefig(f"distances_and_correlations", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')


def show_correlation_distances(datapath: pathlib.Path, analysis_type="full", title="correlation_matrix_distances",
                               annotate_hm=False, save_fig=False):
    cors = {}
    if analysis_type == "drop":
        title = f"{title}_drop"
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            if analysis_type == "drop":
                cors[ds_name] = df[df["entry_type_num"] <= 4][_ACTION_COLUMNS].corr()
            else:
                cors[ds_name] = df[_ACTION_COLUMNS].corr()
    distances = {s: {d: np.linalg.norm(cors[s] - cors[d]) for d in cors.keys()} for s in cors.keys()}
    dist_df = pd.DataFrame(distances)

    pca_df = pd.DataFrame({k: v.to_numpy().flatten() for k, v in cors.items()}, columns=[k for k, _ in cors.items()])
    pca = PCA(n_components=2)
    pca.fit(pca_df)
    plt.cla()
    plt.clf()
    pca_df = pd.DataFrame(pca.components_.T, index=pca_df.columns)
    pca_df = pca_df - pca_df.loc["disability_fest_manual", :]
    pca_df["Corpus"] = ["ASSET", "ASSET", "FestAbility", "UNCRPD", "NewselaManual", "NewselaManual", "NewselaManual",
                        "WikiManual", "WikiManual", "WikiManual", "WikiAuto", "WikiAuto"]
    g = sn.scatterplot(x=pca_df[0], y=pca_df[1], hue=pca_df["Corpus"], s=300, style=pca_df["Corpus"])
    g.set_xlabel("")
    g.set_ylabel("")
    plt.ylim(-0.5, 1.2)
    plt.xlim(-0.05, 0.05)
    for lh in g.legend_.legendHandles:
        lh.set_sizes([100])
    # g.set_xticklabels([])
    # g.set_yticklabels([])
    # plt.grid()
    if analysis_type == "drop":
        plt.title(f"Fine Tuning Correlation Matrices PCA")
    else:
        plt.title(f"Full Corpus Correlation Matrices PCA")

    if save_fig:
        plt.savefig(f"{title}_pca", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')

    dist_df = dist_df.rename(columns=_DS_REMAP_SHORT, index=_DS_REMAP_SHORT)
    dist_df = dist_df.reindex(columns=_DS_ORDER_SHORT, index=_DS_ORDER_SHORT)
    plt.cla()
    plt.clf()
    # sn.heatmap(dist_df, cmap=_CMAP)
    # g = sn.heatmap(scale_df(dist_df), cmap=_CMAP)
    annot_kw = {"fontsize": 7.0, "ma": "center"} if annotate_hm else {}
    g = sn.heatmap(dist_df, cmap=_CMAP,
                   # vmin=0, vmax=2,
                   annot=annotate_hm, annot_kws=annot_kw, fmt=".3f")
    g.set_xticklabels(g.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    # cbar = g.collections[0].colorbar
    # cbar.set_ticks([0, .2, .4, .6, .8, 1])
    # cbar.set_ticklabels(["min", "20%", "40%", "60%", "80%", "max"])
    # sn.clustermap(dist_df, row_cluster=False)
    # plt.title("Operation Correlation Matrix distances", fontsize=15)
    # print(f"{title}: max - {dist_df.max().max():.3f}, min - {dist_df.min().min():.3f}")
    plt.title(title)
    if save_fig:
        plt.savefig(f"{title}.png", bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')


def show_all_action_correlations(datapath: pathlib.Path, save_fig=False):
    count_of_ops_files = 0
    for file in datapath.iterdir():
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            count_of_ops_files += 1
    max_cols = 3
    max_rows = int(count_of_ops_files / max_cols) + 1
    # max_rows = 3
    plt.cla()
    plt.clf()
    fig, axs = plt.subplots(max_rows, max_cols, figsize=set_size(text_width,subplots=(max_rows+4,max_cols+2)))
    plt_row, plt_col = 0, 0
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            add_x = (plt_row == max_rows - 1) #or (plt_row == max_rows - 2 and plt_col == 2)
            sn.heatmap(df[_ACTION_COLUMNS].corr(), ax=axs[plt_row][plt_col], cmap=sn.color_palette("vlag_r", as_cmap=True),
                       vmin=-1, vmax=1, annot=False, cbar=False,
                       xticklabels=add_x, yticklabels=plt_col == 0
                       )
            if plt_col == 0:
                for label in axs[plt_row][plt_col].get_yticklabels():
                    label.set_fontsize(8)
            if plt_row == max_rows-1:
                for label in axs[plt_row][plt_col].get_xticklabels():
                    label.set_fontsize(8)
            axs[plt_row][plt_col].set_title(_DS_REMAP[ds_name], fontsize=10)
            plt_col += 1
            plt_col = plt_col % max_cols
            if plt_col == 0:
                plt_row += 1
            if plt_row > max_rows - 1:
                break
    if plt_row <= max_rows:
        fig.colorbar(axs[2][0].collections[0], cax=axs[plt_row][plt_col])
        for label in axs[plt_row][plt_col].get_yticklabels():
            label.set_fontsize(8)
    plt.suptitle("Action Correlation Matrices")
    if save_fig:
        plt.savefig("all_action_correlations", bbox_inches="tight")
    else:
        plt.show(bbox_inches="tight")
    print()


def process_df(file):
    ds_name = file.stem.split("+")[0]
    df = pd.read_csv(file, sep=';')
    df = df.drop(columns="Unnamed: 0")
    df = df.transpose()
    df = df.rename(columns={0: "sum", 1: "mean", 2: "median", 3: "std"})
    df = df.reset_index()
    df = df.rename(columns={"index": "metric"})
    df.insert(0, "dataset", [ds_name for i in range(len(df))])
    return df, ds_name


def get_joined_metrics(datapath: pathlib.Path, analysis_type: str = "-full"):
    dfs = [process_df(file) for file in datapath.iterdir()
           if file.is_file() and analysis_type in file.stem and file.suffix == ".csv"]
    return pd.concat([df[0] for df in dfs], keys=[df[1] for df in dfs], ignore_index=True)


def show_metric_calculations(data_frame: pd.DataFrame, metric_names: List[str], calc="mean", save_fig=False):
    max_rows = 2
    max_cols = math.ceil(len(metric_names) / 2)
    plt.cla()
    plt.clf()
    fig, axs = plt.subplots(max_rows, max_cols)
    row, col = 0, 0
    for metric in metric_names:
        sn.barplot(data=data_frame[data_frame["metric"] == metric],
                   order=sorted(set(data_frame["dataset"]), key=lambda ds: _DS_ORDER_SHORT.index(ds)),
                   x="dataset",
                   # hue="dataset",
                   y=calc, ax=axs[row][col])
        # plt.xticks(rotation=45,
        #            horizontalalignment='right'
        #            )
        if row == 0:
            axs[row][col].set(xticklabels=[], xlabel=None, ylabel=None)
            axs[row][col].tick_params(bottom=False)
        # if row == 0 and col == 0:
        # axs[row][col].legend()
        else:
            # sorted(distributions.items(), key=lambda pair: _DS_ORDER.index(_DS_REMAP[pair[0]]))
            axs[row][col].set_xticklabels(sorted(set(data_frame["dataset"]), key=lambda ds: _DS_ORDER_SHORT.index(ds)), rotation=45)
            axs[row][col].set(xlabel=None, ylabel=None)
            axs[row][col].tick_params(labelrotation=90)
        axs[row][col].set_title(f"{metric}", fontsize=9)
        col += 1
        col = col % max_cols
        if col == 0:
            row += 1
        if row > max_rows - 1:
            break
    plt.suptitle(f"{calc.capitalize()} values of metrics")
    plt.show()
    print()


def get_align_densities(dataframe: pd.DataFrame):
    result = {k: v for k, v in zip(dataframe["entry_type_num"], dataframe["count"])}
    for i in range(1, 7):
        if i not in result.keys():
            result[i] = 0
    return [result[k] for k in sorted(result.keys())]


def get_align_distance_df(densities, density_type="all"):
    assert density_type == "all" or density_type == "drop" or density_type == "join"
    if density_type == "all":
        distributions = {ds_name: list(np.array(densities[ds_name]) / np.sum(np.array(densities[ds_name])))
                         for ds_name in sorted(densities.keys())}
    elif density_type == "drop":
        distributions = {ds_name: list(np.array(densities[ds_name][:4]) / np.sum(np.array(densities[ds_name][:4])))
                         for ds_name in sorted(densities.keys())}
    else:  # density_type == "join"
        distributions = {}
        for ds_name in sorted(densities.keys()):
            a = np.array([densities[ds_name][0], densities[ds_name][1], densities[ds_name][4],
                          densities[ds_name][2] + densities[ds_name][3] + densities[ds_name][5]])
            distributions[ds_name] = list(a / np.sum(a))
    result_dict = {s: {d: jensen_shannon_distance(distributions[s], distributions[d])
                       for d in distributions.keys()}
                   for s in distributions.keys()}
    return pd.DataFrame(result_dict)


def get_dataset_groups(densities, group_keys=_DS_GROUPS):
    result = {}
    for group, keys in group_keys.items():
        a = np.zeros(len(densities[list(densities.keys())[0]]))
        for k in keys:
            if k in densities:
                a += np.array(densities[k])
        result[group] = list(a)
    return result


def scale_df(dataframe: pd.DataFrame):
    return (dataframe - dataframe.min().min()) / (dataframe.max().max() - dataframe.min().min())


def show_align_distances(datapath: pathlib.Path, group_datasets=False):
    aligns_densities = {}
    for file in tqdm.tqdm(sorted(datapath.iterdir())):
        if file.is_file() and "-entry" in file.stem and file.suffix == ".csv":
            ds_name = file.stem.split("+")[0]
            df = pd.read_csv(file, sep=';')
            aligns_densities[ds_name] = get_align_densities(df)
    aligns_densities["dhcs"] = [1104, 302, 40, 49, 206, 730]
    if group_datasets:
        aligns_densities = get_dataset_groups(aligns_densities)
    for t in ["all", "drop", "join"]:
        result_df = get_align_distance_df(aligns_densities, t)
        result_df = result_df.rename(_DS_REMAP)
        # result_df = pd.DataFrame(_SCALER.fit_transform(result_df), columns=result_df.columns, index=result_df.index)
        # sn.heatmap(result_df, vmin=0, vmax=1, cmap=_CMAP)
        plt.cla()
        plt.clf()
        sn.heatmap(scale_df(result_df), cmap=_CMAP)
        plt.title(f"Jensen-Shannon distance of Alignment Type Distributions ({t})", fontsize=15)
        plt.show()
    print()


# def fix_add_del(datapath: pathlib.Path):  # TO-DO: Remove - fixed in pipeline
#     for file in tqdm.tqdm(sorted(datapath.iterdir())):
#         if file.is_file() and "-ops" in file.stem and file.suffix == ".csv":
#             ds_name = file.stem.split("+")[0]
#             df = pd.read_csv(file, sep=';')
#             df["ADD"][df["entry_type_num"] == 5] = 0
#             df.to_csv(f"{data_path}/{file.stem}+fixed.csv", sep=';')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/dataset_analysis/csvs")
    subparsers = parser.add_subparsers(dest="analyses_to_perform")
    all_parser = subparsers.add_parser("all", aliases=["a"], help="Perform all analyses")
    set_option_parser = subparsers.add_parser("set", aliases=["s"])
    set_option_parser.add_argument("--distances_and_correlations", "-dc", action='store_true')
    set_option_parser.add_argument("--histograms", "-hs", action='store_true')
    set_option_parser.add_argument("--distances", "-d", action='store_true')
    set_option_parser.add_argument("--correlation_distances", "-c", action='store_true')
    set_option_parser.add_argument("--metrics", "-m", action='store_true')
    set_option_parser.add_argument("--action_correlations", "-ac", action='store_true')
    set_option_parser.add_argument("--correlation_matrices", "-cm", action='store_true')
    parser.add_argument("--analysis_type", "-at", choices=["full","drop"], default="full")
    parser.add_argument("--save_fig", action="store_true")

    args = parser.parse_args()

    if args.analyses_to_perform is None or args.analyses_to_perform == "all":
        args.distances_and_correlations = True
        args.histograms = True
        args.distances = True
        args.correlation_distances = True
        args.metrics = True
        args.action_correlations = True
        args.correlation_matrices = True

    data_path = pathlib.Path(args.data_path)

    if args.distances_and_correlations:
        show_dist_and_corr_together(data_path, analysis_type=args.analysis_type, save_fig=args.save_fig)

    if args.histograms:
        show_action_histograms(data_path, analysis_type=args.analysis_type, save_fig=args.save_fig)

    if args.distances:
        show_action_distribution_distances(data_path, title="JSD_action_distances", save_fig=args.save_fig,
                                           annotate_hm=True, analysis_type=args.analysis_type)

    # TO-DO: maybe re-add other metrics
    # show_action_distribution_distances(data_path, series_norm, title="NORM_action_distances_fixed", analysis_type="drop", annotate_hm=True)
    # show_action_distribution_distances(data_path, sum_pointwise_jsd, title="SUM_JSD_action_distances_fixed", analysis_type="drop", annotate_hm=True)

    if args.correlation_distances:
        show_correlation_distances(data_path, title="correlation_matrix_distances_fixed", save_fig=args.save_fig,
                                   analysis_type=args.analysis_type, annotate_hm=True)


    # show_align_distances(data_path)
    # show_align_distances(data_path, group_datasets=True)

    if args.metrics:
        cc = get_joined_metrics(data_path, "-no-5-6")
        cc = cc.replace({"dataset": _DS_REMAP_SHORT})
        # print(cc)
        show_metric_calculations(cc, _NUMERIC)
        # show_metric_calculations(cc, _PARAPHRASE)
        # show_metric_calculations(cc, _OTHER)

    if args.correlation_matrices:
        show_all_action_correlations(data_path)
    # show_all_action_correlations(data_path / "newsela_levels")
    # show_all_action_correlations(data_path / "asset_annotators")
