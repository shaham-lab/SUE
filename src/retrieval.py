import sys
import json
import warnings
import argparse

from data import *
from trainer import *
from general_utils import *


def load_config_file(dataset_name):
    with open(f"../configs/{dataset_name}_config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def main():
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SUE training and testing')
    parser.add_argument('data', type=str, help='Dataset name (e.g., flickr30)')
    parser.add_argument('--train', action='store_true', help='Perform training')
    parser.add_argument('--test', action='store_true', help='Perform testing')
    args = parser.parse_args()
    
    dataset_name = args.data
    configs = load_config_file(dataset_name)
    train_set, test_set = load_dataset(dataset_name, n_test=configs["n_test"])
    train_set = create_weakly_parallel_data(train_set, n_parallel=configs["n_parallel"])
    
    trainer = Trainer(
        dataset_name=dataset_name,
        n_parallel=configs["n_parallel"], 
        n_components=configs["n_components"],
        configs=configs
    )
    
    checkpoint_path = f'../checkpoints/checkpoints_{dataset_name}.pth'
    
    if args.train:
        trainer.fit(train_set=train_set, with_cca=True, with_mmd=True, with_se=True)
        
    if args.test or not args.train:  # Default to test if no flags provided
        load_checkpoint(trainer, checkpoint_path)
        
    trainer.test(test_set=test_set)

if __name__ == "__main__":
    main()
