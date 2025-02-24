import sys
import json
import warnings

from data import *
from trainer import *
from general_utils import *





def load_config_file(dataset_name):
    with open(f"../configs/{dataset_name}_config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def main():

    warnings.filterwarnings("ignore", category=UserWarning)

    dataset_name = sys.argv[1]
    configs = load_config_file(dataset_name)
    train_set, test_set = load_dataset(dataset_name, n_test=configs["n_test"])
    train_set = create_weakly_parallel_data(train_set, n_parallel=configs["n_parallel"])


    trainer = Trainer(
        dataset_name=dataset_name,
        n_parallel=configs["n_parallel"], 
        n_eigenvectors=configs["n_eigenvectors"], 
        n_components=configs["n_components"],
        configs=configs
    )

    # trainer.fit(train_set=train_set, with_cca=True, with_mmd=True, with_se=True)
    # save_checkpoint(trainer, f'../checkpoints/checkpoints_{dataset_name}.pth')
    load_checkpoint(trainer, f'../checkpoints/checkpoints_{dataset_name}.pth')
    trainer.test(test_set=test_set)




if __name__ == "__main__":
    main()
