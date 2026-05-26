import argparse
import os
from utils.data_utils import create_and_save_splits_joint, load_dataset

def prepare_and_save_splits(
    dataset_name,
    strategy,
    train_ratio,
    num_splits,
    base_dir='data',
    max_neighbors=9
):
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Construct split path
    split_path = os.path.join(base_dir, f'{dataset_name}_{strategy}_{train_ratio}_{max_neighbors}')
    
    print(f"Saving splits to: {split_path}")

    # Save splits
    create_and_save_splits_joint(
        dataset,
        save_dir=split_path,
        train_ratio=train_ratio,
        strategy=strategy,
        num_splits=num_splits,
        max_neighbors=max_neighbors
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and save splits for a dataset.')
    parser.add_argument('--dataset_name', type=str, default='cora', help='Name of the dataset')
    parser.add_argument('--strategy', type=str, default='random', choices=['random', 'snowball'], help='Splitting strategy')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Train ratio for random splitting')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of splits to generate')
    
    args = parser.parse_args()

    prepare_and_save_splits(
        dataset_name=args.dataset_name,
        strategy=args.strategy,
        train_ratio=args.train_ratio,
        num_splits=args.num_splits,
        max_neighbors=9  # Default value, can be adjusted as needed
    )


