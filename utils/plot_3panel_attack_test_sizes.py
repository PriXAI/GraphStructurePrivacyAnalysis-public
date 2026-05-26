import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "font.size": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DEFAULT_CSVS = [
    "results/graph_structure_privacy_results_cora.csv",
    "results/graph_structure_privacy_results_chameleon.csv",
    "results/graph_structure_privacy_results_pubmed.csv",
]

DATASET_ORDER = ["cora", "chameleon", "pubmed"]
DATASET_DISPLAY_NAMES = {
    "cora": "Cora",
    "chameleon": "Chameleon",
    "pubmed": "Pubmed",
}
MODEL_ORDER = ["GCN", "GraphSage", "GAT"]
MODEL_NAME_MAP = {
    "GraphSAGE": "GraphSage",
}
ACCESS_SPECS = [
    ("Original Split", "gen_gap_orig", "ma_orig", "#1f77b4"),
    ("Full Graph", "gen_gap_alledges", "ma_transductive", "#ff7f0e"),
    ("No Edges", "gen_gap_noedges", "ma_nograph", "#2ca02c"),
]


def parse_mean_std(value):
    text = str(value).strip().replace("\\pm", "±").replace("+/-", "±")
    match = re.match(r"^\s*([-+]?\d*\.?\d+)\s*±\s*([-+]?\d*\.?\d+)\s*$", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float(text), 0.0


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def normalize_model_name(model_name):
    return MODEL_NAME_MAP.get(model_name, model_name)


def ordered_values(values, preferred_order):
    value_set = set(values)
    ordered = [value for value in preferred_order if value in value_set]
    ordered.extend(sorted(value for value in value_set if value not in preferred_order))
    return ordered


def load_plot_frame(csv_paths, sampling, use_loss):
    frames = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["Use_Loss"] = df["Use_Loss"].map(normalize_bool)
        df["Attack Test Size"] = df["Attack Test Size"].astype(float)
        df["Train Ratio"] = df["Train Ratio"].astype(float)
        df["Model_type"] = df["Model_type"].map(normalize_model_name)
        df = df[(df["Use_Loss"] == use_loss) & (df["Strategy"] == sampling)].copy()

        records = []
        for _, row in df.iterrows():
            for access_name, gap_col, adv_col, color in ACCESS_SPECS:
                gap_mean, gap_std = parse_mean_std(row[gap_col])
                adv_mean, adv_std = parse_mean_std(row[adv_col])
                records.append(
                    {
                        "Dataset Name": row["Dataset Name"],
                        "Model_type": row["Model_type"],
                        "Train Ratio": float(row["Train Ratio"]),
                        "Attack Test Size": float(row["Attack Test Size"]),
                        "Graph Access": access_name,
                        "Color": color,
                        "Performance Gap Mean": gap_mean,
                        "Performance Gap STD": gap_std,
                        "Membership Advantage Mean": adv_mean,
                        "Membership Advantage STD": adv_std,
                    }
                )

        frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame()

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df = plot_df.sort_values(
        by=["Dataset Name", "Graph Access", "Model_type", "Train Ratio", "Attack Test Size"]
    )
    return plot_df


def plot_three_panel_attack_test_sizes(plot_df, sampling, output_path):
    if plot_df.empty:
        raise ValueError("No rows found after filtering the results files.")

    datasets = ordered_values(plot_df["Dataset Name"].unique(), DATASET_ORDER)
    attack_sizes = sorted(plot_df["Attack Test Size"].unique())

    fig, axes = plt.subplots(
        len(attack_sizes),
        len(datasets),
        figsize=(4.1 * len(datasets), 2.8 * len(attack_sizes)),
        sharey=True,
        squeeze=False,
    )

    dataset_limits = {}
    for dataset_name in datasets:
        dataset_df = plot_df[plot_df["Dataset Name"] == dataset_name].copy()
        x_min = (dataset_df["Performance Gap Mean"] - dataset_df["Performance Gap STD"]).min()
        x_max = (dataset_df["Performance Gap Mean"] + dataset_df["Performance Gap STD"]).max()
        x_pad = max(0.8, 0.08 * (x_max - x_min if x_max > x_min else 1.0))
        dataset_limits[dataset_name] = (x_min - x_pad, x_max + x_pad)

    y_max = min(
        1.02,
        max(
            1.0,
            plot_df["Membership Advantage Mean"].max()
            + plot_df["Membership Advantage STD"].max()
            + 0.03,
        ),
    )

    for row_idx, attack_size in enumerate(attack_sizes):
        for col_idx, dataset_name in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            panel_df = plot_df[
                (plot_df["Dataset Name"] == dataset_name)
                & (plot_df["Attack Test Size"] == attack_size)
            ].copy()

            for access_name, _, _, color in ACCESS_SPECS:
                access_df = panel_df[panel_df["Graph Access"] == access_name]
                if access_df.empty:
                    continue

                ax.errorbar(
                    access_df["Performance Gap Mean"],
                    access_df["Membership Advantage Mean"],
                    xerr=access_df["Performance Gap STD"],
                    yerr=access_df["Membership Advantage STD"],
                    fmt="o",
                    linestyle="none",
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=5.8,
                    elinewidth=1.0,
                    capsize=2.2,
                    alpha=0.78,
                )

            ax.set_xlim(*dataset_limits[dataset_name])
            ax.set_ylim(0.0, y_max)
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.5)

            if row_idx == 0:
                ax.set_title(DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.capitalize()))

            if row_idx == len(attack_sizes) - 1:
                ax.set_xlabel("Performance gap (%)")

    fig.suptitle(f"{sampling.capitalize()} sampling", y=0.958)
    fig.supylabel("Membership advantage", x=0.031)

    access_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="none",
            markersize=8,
            label=access_name,
        )
        for access_name, _, _, color in ACCESS_SPECS
    ]
    fig.legend(
        handles=access_handles,
        loc="lower center",
        ncol=3,
        title="Graph Access",
        bbox_to_anchor=(0.5, 0.03),
        frameon=True,
        fancybox=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.045, 0.10, 1, 0.985], h_pad=0.35, w_pad=0.3)
    for row_idx, attack_size in enumerate(attack_sizes):
        row_box = axes[row_idx][0].get_position()
        y_center = 0.5 * (row_box.y0 + row_box.y1)
        r_value = 1.0 - attack_size
        fig.text(
            0.075,
            y_center,
            f"r={r_value:g}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.5,
        )
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def default_output_path(sampling, suffix):
    return Path("plots") / f"three_panel_perf_vs_adv_attack_test_sizes_{sampling}.{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance gap vs membership advantage across attack test sizes."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=DEFAULT_CSVS,
        help="Result CSVs to include",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="snowball",
        choices=["random", "snowball"],
        help="Sampling strategy to plot",
    )
    parser.add_argument(
        "--use_loss",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Filter rows by Use_Loss",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path",
    )
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = default_output_path(args.sampling, "png")

    plot_df = load_plot_frame(args.csv, args.sampling, args.use_loss)
    plot_three_panel_attack_test_sizes(plot_df, args.sampling, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
