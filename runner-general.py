from main import main as run_experiment

datasets = ["cora", "pubmed", "chameleon"]
models = ["GCN", "GAT", "GraphSage"]
train_ratios = [0.1, 0.5]
strategies = ["snowball", "random"]

num_splits = 5
max_neighbors = 3
random_seed = 42
attack_test_size = [0.9, 0.8, 0.5, 0.2]

for dataset in datasets:
    for model in models:
        for train_ratio in train_ratios:
            for strategy in strategies:
                if dataset == "chameleon":
                    data_dir = "data/chameleon1"
                else:
                    data_dir = f"data/{dataset}"

                if strategy == "snowball":
                    input_path = f"{data_dir}/snowball_{max_neighbors}_{train_ratio}"
                else:
                    input_path = f"{data_dir}/random_{train_ratio}"

                output_path = f"results/graph_structure_privacy_results_{dataset}.csv"

                run_experiment(
                    dataset_name=dataset,
                    model_type=model,
                    train_ratio=train_ratio,
                    num_splits=num_splits,
                    max_neighbors=max_neighbors,
                    random_seed=random_seed,
                    strategy=strategy,
                    input_path=input_path,
                    output_path=output_path,
                    attack_test_size=attack_test_size,
                )
