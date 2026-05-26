
import utils.data_utils as ut
import attacks.mi_attack as att
import attacks.attack_utils as att_ut
import models.model as md
import training.train as tr

import torch
import training.evaluate as eval
import random
import numpy as np
import csv
import argparse
import os

CURRENT_BASIC_COLUMNS = [
    'Dataset Name',
    'Model_type',
    'Train Ratio',
    'Train Size',
    'Train Size Min',
    'Train Size Max',
    'Num Splits',
    'Max_neighbors',
    'Strategy',
    'Use_Loss',
    'Attack Test Size',
]

def build_results_template():
    return {
        'avgdegree_train': [],
        'avgdegree_test': [],
        'train_acc_orig': [],
        'test_acc_orig': [],
        'gen_gap_orig': [],
        'ma_orig': [],
        'ma_transductive': [],
        'ma_nograph': [],

        'train_acc_alledges': [],
        'test_acc_alledges': [],
        'gen_gap_alledges': [],

        'train_acc_noedges': [],
        'test_acc_noedges': [],
        'gen_gap_noedges': [],

        'attack_acc_orig': [],
        'attack_auc_orig': [],
        'attack_ap_orig': [],

        'attack_acc_nograph': [],
        'attack_auc_nograph': [],
        'attack_ap_nograph': [],

        'attack_acc_transductive': [],
        'attack_auc_transductive': [],
        'attack_ap_transductive': [],

        'precision_orig': [],
        'f1_orig': [],
        'precision_nograph': [],
        'f1_nograph': [],
        'precision_transductive': [],
        'f1_transductive': [],
    }

def build_results_header():
    return CURRENT_BASIC_COLUMNS + list(build_results_template().keys())

def summarize_train_sizes(train_sizes):
    if not train_sizes:
        raise ValueError("No train sizes were recorded. At least one split is required.")

    unique_sizes = sorted({int(size) for size in train_sizes})
    train_size_min = unique_sizes[0]
    train_size_max = unique_sizes[-1]

    if len(unique_sizes) == 1:
        return train_size_min, train_size_min, train_size_max

    train_size_mean = float(np.mean(train_sizes))
    print(
        f"Warning: Train size varied across splits {unique_sizes}; "
        "recording mean/min/max in the CSV metadata."
    )

    if train_size_mean.is_integer():
        train_size_mean = int(train_size_mean)

    return train_size_mean, train_size_min, train_size_max

def ensure_output_path_exists(output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def validate_results_csv_header(output_path):
    expected_header = build_results_header()
    with open(output_path, newline='') as file:
        reader = csv.DictReader(file)
        existing_header = reader.fieldnames or []

        if existing_header == expected_header:
            return

        raise ValueError(
            "Existing results CSV header does not match the current format. "
            f"Expected {expected_header}, but found {existing_header}. "
            f"Remove or rename {output_path} if you want to regenerate it."
        )

def normalize_attack_test_sizes(attack_test_size):
    if isinstance(attack_test_size, np.ndarray):
        sizes = attack_test_size.tolist()
    elif isinstance(attack_test_size, (list, tuple)):
        sizes = list(attack_test_size)
    else:
        sizes = [attack_test_size]

    normalized_sizes = []
    seen_sizes = set()
    for size in sizes:
        size = float(size)
        if size not in seen_sizes:
            normalized_sizes.append(size)
            seen_sizes.add(size)

    return normalized_sizes

def main(
    dataset_name,
    model_type,
    train_ratio,
    num_splits,
    max_neighbors,
    random_seed,
    strategy,
    input_path,
    output_path,
    attack_test_size,
):
    attack_test_sizes = normalize_attack_test_sizes(attack_test_size)
    attack_uses_loss = True

    # Set a fixed random seed for reproducibility
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #Load data
    if dataset_name == "chameleon":
       data=torch.load("data/chameleon")
    else:
        data=ut.load_dataset(dataset_name)

    num_nodes=data.x.shape[0]

    results_by_attack_test_size = {
        size: build_results_template() for size in attack_test_sizes
    }
    observed_train_sizes = []


    # For each split, we train the model and compute the train and test accuracies
    for split in range(num_splits):
        print(f"Running experiment on split {split + 1}/{num_splits}")
        split_path = f'{input_path}/split_{split+1}.pt'
        
        joint_data = torch.load(split_path)

       
        train_indices = joint_data.train_mask.nonzero(as_tuple=True)[0]
        test_indices = joint_data.test_mask.nonzero(as_tuple=True)[0]
        attack_labels = att_ut.construct_attack_labels(data.num_nodes, train_indices, test_indices)
        train_size = joint_data.train_mask.sum().item()
        observed_train_sizes.append(train_size)

        #average degree of the train data
        avgdegree_train=joint_data.train_edge_mask.sum().item()/(train_size)
        avgdegree_test=joint_data.test_edge_mask.sum().item()/((num_nodes-train_size))

        # ----- START training for regular models -------------- 
        target_gnn = md.GNNModel(
            in_channels=data.x.shape[1],
            hidden_channels=64,
            out_channels=len(data.y.unique()),
            model_type=model_type
        )

        optimizer = torch.optim.Adam(target_gnn.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(200):
            loss = tr.train(target_gnn, joint_data, optimizer)
        orig_posteriors=eval.get_posterior(target_gnn, joint_data)
        # ----- END training for regular models -------------- 

        
        transductive_posteriors=eval.get_posterior(target_gnn,data)
        nograph_posteriors=eval.compute_posterior_without_edges(target_gnn, data)

        train_acc,test_acc = eval.compute_accuracy(orig_posteriors, joint_data)
        train_acc_orig = train_acc
        test_acc_orig = test_acc
        gen_gap_orig = (train_acc-test_acc)*100/train_acc

        train_acc,test_acc = eval.compute_accuracy(transductive_posteriors, joint_data)
        train_acc_alledges = train_acc
        test_acc_alledges = test_acc
        gen_gap_alledges = (train_acc-test_acc)*100/train_acc
       
        train_acc,test_acc = eval.compute_accuracy(nograph_posteriors, joint_data)
        train_acc_noedges = train_acc
        test_acc_noedges = test_acc
        gen_gap_noedges = (train_acc-test_acc)*100/train_acc


        for current_attack_test_size in attack_test_sizes:
            results = results_by_attack_test_size[current_attack_test_size]
            results['avgdegree_train'].append(avgdegree_train)
            results['avgdegree_test'].append(avgdegree_test)

            results['train_acc_orig'].append(train_acc_orig)
            results['test_acc_orig'].append(test_acc_orig)
            results['gen_gap_orig'].append(gen_gap_orig)

            results['train_acc_alledges'].append(train_acc_alledges)
            results['test_acc_alledges'].append(test_acc_alledges)
            results['gen_gap_alledges'].append(gen_gap_alledges)

            results['train_acc_noedges'].append(train_acc_noedges)
            results['test_acc_noedges'].append(test_acc_noedges)
            results['gen_gap_noedges'].append(gen_gap_noedges)

            # Create 3 random splits for the current attack test size
            node_indices = torch.arange(data.num_nodes)
            splits_attack = att_ut.create_random_splits(
                node_indices,
                3,
                test_size=current_attack_test_size,
                seed=1000,
                stratify_labels=attack_labels,
            )
        
            # Run attack model training for trained GNN posteriors
            print(f"Running attack for trained GNN posteriors with attack_test_size={current_attack_test_size}...")

            acc, auc,ap, precision, f1, membership_advantage = att.run_attack_for_splits(
                splits_attack,
                orig_posteriors,
                data.y,
                train_indices,
                test_indices,
                num_epochs=200,
            )
            results['attack_acc_orig'].append(acc)
            results['attack_auc_orig'].append(auc)
            results['attack_ap_orig'].append(ap)
            results['precision_orig'].append(precision)
            results['ma_orig'].append(membership_advantage)
            results['f1_orig'].append(f1)

            acc, auc,ap, precision,f1, membership_advantage = att.run_attack_for_splits(
                splits_attack,
                transductive_posteriors,
                data.y,
                train_indices,
                test_indices,
                num_epochs=200,
            )
            results['attack_acc_transductive'].append(acc)
            results['attack_auc_transductive'].append(auc)
            results['attack_ap_transductive'].append(ap)
            results['precision_transductive'].append(precision)
            results['ma_transductive'].append(membership_advantage)
            results['f1_transductive'].append(f1)

            acc, auc,ap, precision,f1, membership_advantage = att.run_attack_for_splits(
                splits_attack,
                nograph_posteriors,
                data.y,
                train_indices,
                test_indices,
                num_epochs=200,
            )
            results['attack_acc_nograph'].append(acc)
            results['attack_auc_nograph'].append(auc)
            results['attack_ap_nograph'].append(ap)
            results['precision_nograph'].append(precision)
            results['ma_nograph'].append(membership_advantage)
            results['f1_nograph'].append(f1)
        
        
    
    # Check if the file exists and is not empty
    ensure_output_path_exists(output_path)
    file_exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    header = build_results_header()
    train_size_value, train_size_min, train_size_max = summarize_train_sizes(observed_train_sizes)

    if file_exists:
        validate_results_csv_header(output_path)

    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
          # If the file does not exist or is empty, write the header  
        if not file_exists:
            writer.writerow(header)
        # Each row contains the basic information + the mean and stdev of the results
        for current_attack_test_size in attack_test_sizes:
            results = results_by_attack_test_size[current_attack_test_size]
            row = [
                dataset_name,
                model_type,
                train_ratio,
                train_size_value,
                train_size_min,
                train_size_max,
                len(observed_train_sizes),
                max_neighbors,
                strategy,
                attack_uses_loss,
                current_attack_test_size,
            ]
            for key in results.keys():
                row.append(f'{np.mean(results[key]):.4f} \\pm {np.std(results[key]):.4f}')
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph-structure privacy analysis experiments")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model (e.g., GAT)")
    parser.add_argument("--train_ratio", type=float, required=True, help="Training ratio")
    parser.add_argument("--num_splits", type=int, required=True, help="Number of splits")
    parser.add_argument("--max_neighbors", type=int, default=3, help="Maximum number of sampled neighbors")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--strategy", type=str, default='snowball', help="Sampling label written to the CSV output")
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing the saved split files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--attack_test_size", type=float, nargs='+', default=[0.2], help="one or more test sizes for the attack model")
    args = parser.parse_args()
    main(
        args.dataset_name,
        args.model_type,
        args.train_ratio,
        args.num_splits,
        args.max_neighbors,
        args.random_seed,
        args.strategy,
        args.input_path,
        args.output_path,
        args.attack_test_size,
    )
