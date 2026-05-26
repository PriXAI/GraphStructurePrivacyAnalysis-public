import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd


MODEL_ORDER = ["GCN", "GraphSage", "GAT"]
MODEL_NAME_MAP = {
    "GraphSAGE": "GraphSage",
}
SETTING_ORDER = ["orig", "alledges", "noedges"]
SETTING_LABELS = {
    "orig": "Orig",
    "alledges": "FullGraph",
    "noedges": "NoEdges",
}
SETTING_COLORS = {
    "orig": "#F4A261",
    "alledges": "#2A9D8F",
    "noedges": "#E76F51",
}
STRATEGY_ORDER = ["random", "snowball"]
STRATEGY_LABELS = {"random": "random", "snowball": "snowball"}


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


def choose_y_tick_step(y_data_max, approx_intervals=8):
    if y_data_max <= 0:
        return 1.0

    raw_step = y_data_max / float(approx_intervals)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    candidates = [1.0, 2.0, 2.5, 5.0, 10.0]
    normalized = raw_step / magnitude
    best_base = min(candidates, key=lambda base: abs(base - normalized))
    return best_base * magnitude


def y_tick_formatter(value, _pos):
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:g}"


def build_y_axis(y_data_max):
    step = choose_y_tick_step(y_data_max)
    n_intervals = int(math.floor(y_data_max / step)) + 1
    y_max = n_intervals * step
    y_ticks = [idx * step for idx in range(n_intervals + 1)]
    return y_max, y_ticks


def build_plot_frame(df, dataset_name, use_loss, attack_test_size, max_neighbors):
    filtered = df.copy()
    filtered["Use_Loss"] = filtered["Use_Loss"].map(normalize_bool)
    filtered["Train Ratio"] = filtered["Train Ratio"].astype(float)
    filtered["Attack Test Size"] = filtered["Attack Test Size"].astype(float)
    filtered["Max_neighbors"] = filtered["Max_neighbors"].astype(int)
    filtered["Model_type"] = filtered["Model_type"].map(normalize_model_name)

    filtered = filtered[filtered["Dataset Name"] == dataset_name]
    filtered = filtered[filtered["Use_Loss"] == use_loss]
    filtered = filtered[filtered["Attack Test Size"] == float(attack_test_size)]
    filtered = filtered[filtered["Max_neighbors"] == int(max_neighbors)]

    records = []
    for _, row in filtered.iterrows():
        for setting in SETTING_ORDER:
            mean, std = parse_mean_std(row[f"gen_gap_{setting}"])
            records.append(
                {
                    "Model_type": row["Model_type"],
                    "Train Ratio": float(row["Train Ratio"]),
                    "Train Size": int(row["Train Size"]),
                    "Strategy": row["Strategy"],
                    "Setting": setting,
                    "Mean": mean,
                    "STD": std,
                }
            )

    return pd.DataFrame(records).drop_duplicates(
        subset=["Model_type", "Train Ratio", "Train Size", "Strategy", "Setting"],
        keep="first",
    )


def plot_gengap_sampling_strategy(
    df, dataset_name, use_loss, attack_test_size, max_neighbors, output_path
):
    plot_df = build_plot_frame(df, dataset_name, use_loss, attack_test_size, max_neighbors)
    if plot_df.empty:
        raise ValueError(
            f"No rows found for dataset='{dataset_name}', use_loss={use_loss}, "
            f"attack_test_size={attack_test_size}, max_neighbors={max_neighbors}."
        )

    models = ordered_values(plot_df["Model_type"].unique(), MODEL_ORDER)
    train_ratios = sorted(plot_df["Train Ratio"].unique())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        len(models),
        len(train_ratios),
        figsize=(3.8 * len(train_ratios), 3.2 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_positions = {strategy: idx for idx, strategy in enumerate(STRATEGY_ORDER)}
    bar_width = 0.18
    bar_offsets = {
        "orig": -0.27,
        "alledges": 0.0,
        "noedges": 0.27,
    }
    y_data_max = float((plot_df["Mean"] + plot_df["STD"]).max())
    y_max, y_ticks = build_y_axis(y_data_max)

    for row_idx, model in enumerate(models):
        for col_idx, train_ratio in enumerate(train_ratios):
            ax = axes[row_idx][col_idx]
            panel = plot_df[
                (plot_df["Model_type"] == model)
                & (plot_df["Train Ratio"] == train_ratio)
            ]

            for strategy in STRATEGY_ORDER:
                strategy_panel = panel[panel["Strategy"] == strategy]
                if strategy_panel.empty:
                    continue

                for setting in SETTING_ORDER:
                    point = strategy_panel[strategy_panel["Setting"] == setting]
                    if point.empty:
                        continue

                    point = point.iloc[0]
                    xpos = x_positions[strategy] + bar_offsets[setting]
                    ax.bar(
                        xpos,
                        point["Mean"],
                        yerr=point["STD"],
                        width=bar_width,
                        capsize=3,
                        color=SETTING_COLORS[setting],
                        hatch="" if setting == "orig" else ("//" if setting == "alledges" else "xx"),
                        edgecolor="black",
                        linewidth=0.6,
                    )

            if col_idx == 0:
                ax.set_ylabel(f"{model}\nPerformance Gap")
            if row_idx == 0:
                ax.set_title(f"{dataset_name} — Train = {train_ratio * 100:.1f}%")

            ax.set_xticks([0, 1])
            ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlim(-0.45, 1.45)
            ax.set_ylim(0, y_max)
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_tick_formatter))
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

            if row_idx == len(models) - 1:
                ax.set_xlabel("Sampling Strategy")

    legend_handles = [
        mpatches.Patch(
            facecolor=SETTING_COLORS[setting],
            hatch="" if setting == "orig" else ("//" if setting == "alledges" else "xx"),
            edgecolor="black",
            label=SETTING_LABELS[setting],
        )
        for setting in SETTING_ORDER
    ]
    fig.legend(
        legend_handles,
        [SETTING_LABELS[s] for s in SETTING_ORDER],
        title="Graph Access",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def default_output_path(dataset_name):
    return Path("plots") / f"{dataset_name}_gengap_sampling_strategy.png"


def main():
    parser = argparse.ArgumentParser(
        description="Plot generalization gap changes across sampling strategies."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to results CSV")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--use_loss",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Filter rows by Use_Loss",
    )
    parser.add_argument(
        "--attack_test_size",
        type=float,
        default=0.2,
        help="Attack test size to fix in the plot",
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=3,
        help="Filter rows by Max_neighbors",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_path = Path(args.output) if args.output else default_output_path(args.dataset)
    plot_gengap_sampling_strategy(
        df=df,
        dataset_name=args.dataset,
        use_loss=args.use_loss,
        attack_test_size=args.attack_test_size,
        max_neighbors=args.max_neighbors,
        output_path=output_path,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
