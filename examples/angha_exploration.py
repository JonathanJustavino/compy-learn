import gc
import os
import tqdm
import json
import shutil
import datetime
from GPUtil import showUtilization as gpu_usage
import numpy as np
import torch.utils.data

from torch import nn

from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver
from compy.datasets.anghabench_graph import AnghabenchGraphDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from compy.datasets import dataflow_preprocess
from compy.datasets.dataflow_preprocess import main as dataflow_main


# FLAGS = flags.FLAGS
# FLAGS.out_dir = dataset_path
# FLAGS.preprocess = True
# FLAGS.eliminate_data_duplicates = True
# FLAGS.debug = False

def accumulate_branch_count(weights, dataloader, max_branch_length):
    total_branch_count = 0
    for sample in tqdm.tqdm(dataloader):
        sample = sample.cuda()
        target = sample.y * 100
        target = target.int()
        train_bincount = torch.bincount(target.view(-1), minlength=max_branch_length).int()
        weights += train_bincount
        total_branch_count += len(sample.y)
    return total_branch_count, weights


def recompute_branch_weights(training_subset, validation_subset, test_subset, batch_size=512, num_workers=3):
    max_branch_length = 101
    training_weights = torch.zeros(max_branch_length).cuda()
    validation_weights = torch.zeros(max_branch_length).cuda()
    testing_weights = torch.zeros(max_branch_length).cuda()

    train_loader = GeometricDataLoader(training_subset, batch_size=batch_size, num_workers=num_workers)
    valid_loader = GeometricDataLoader(validation_subset, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_subset, batch_size=batch_size, num_workers=num_workers)

    if not torch.cuda.is_available():
        print("GPU not available")
        exit()

    print("Computing amount of train and test branches")
    train_branch_count, training_weights = accumulate_branch_count(training_weights, train_loader, max_branch_length)
    valid_branch_count, validation_weights = accumulate_branch_count(validation_weights, valid_loader, max_branch_length)
    test_branch_count, testing_weights = accumulate_branch_count(testing_weights, test_loader, max_branch_length)

    total_train_probabilities = sum(training_weights).int().item()
    total_valid_probabilities = sum(validation_weights).int().item()
    total_test_probabilities = sum(testing_weights).int().item()

    training_weights = training_weights.div(total_train_probabilities)
    validation_weights = training_weights.div(total_valid_probabilities)
    testing_weights = testing_weights.div(total_test_probabilities)
    classes = [0, 3, 37, 50, 62, 96, 100]
    class_count = len(classes)

    for index in classes:
        training_weights[index] = 1 / class_count
        testing_weights[index] = 1 / class_count
        validation_weights[index] = 1 / class_count

    total_train_branches = round(total_train_probabilities / 2)
    total_test_branches = round(total_test_probabilities / 2)
    total_valid_branches = round(total_test_probabilities / 2)

    return training_weights, validation_weights, testing_weights, total_train_branches, total_valid_branches, total_test_branches


def lookup_branch_weights(training_subset, validation_subset, test_subset, dataset, cache_path="cache/cached_weights.pt", recompute=False):
    cache_path = f"{dataset.dataset_info_path}/{cache_path}"
    cache_exists = os.path.exists(cache_path)
    train_indices = []
    test_indices = []

    if cache_exists and not recompute:
        print("Using cached weights")
        cached_weights = torch.load(cache_path)
        train_weights = cached_weights["train_weights"]
        valid_weights = cached_weights["valid_weights"]
        test_weights = cached_weights["test_weights"]
        train_indices = cached_weights["train_indices"]
        valid_indices = cached_weights["valid_indices"]
        test_indices = cached_weights["test_indices"]
        total_train_branches = cached_weights["total_train_branches"]
        total_valid_branches = cached_weights["total_valid_branches"]
        total_test_branches = cached_weights["total_test_branches"]

    if recompute or not (train_indices == training_subset.indices and test_indices == test_subset.indices and valid_indices == validation_subset.indices):
        train_weights, valid_weights, test_weights, total_train_branches, total_valid_branches, total_test_branches = recompute_branch_weights(training_subset, validation_subset, test_subset, batch_size=512, num_workers=3)

        print("Caching results")
        new_weights = {
            'train_weights': train_weights,
            'valid_weights': valid_weights,
            'test_weights': test_weights,
            'train_indices': training_subset.indices,
            'valid_indices': validation_subset.indices,
            'test_indices': test_subset.indices,
            'total_train_branches': total_train_branches,
            'total_valid_branches': total_valid_branches,
            'total_test_branches': total_test_branches
        }

        torch.save(new_weights, cache_path)

    setattr(training_subset, 'total_branches', total_train_branches)
    setattr(validation_subset, 'total_branches', total_valid_branches)
    setattr(test_subset, 'total_branches', total_test_branches)

    print(f"Amount of training branches {total_train_branches}\n"
          f"Amount of valid branches {total_valid_branches}\n"
          f"Amount of test branches {total_test_branches}")

    return train_weights, valid_weights, test_weights


def split_dataset(dataset):
    amount_samples = dataset.total_num_samples
    amount_samples = amount_samples // 900
    test_range = round(float(amount_samples) * 0.1)
    valid_range = round(float(amount_samples) * 0.1)
    train_range = round(amount_samples - (test_range + valid_range))

    train_samples = [index for index in range(0, train_range)]
    validation_samples = [index for index in range(0, valid_range)]
    test_samples = [index for index in range(train_range, amount_samples)]

    training_set = torch.utils.data.Subset(dataset, train_samples)
    validation_set = torch.utils.data.Subset(dataset, validation_samples)
    test_set = torch.utils.data.Subset(dataset, test_samples)

    return training_set, validation_set, test_set


def move_results(training_model, exploration_path, configurations_length, model_config):
    train_weights = model_config["train_weights"].cpu().numpy()
    valid_weights = model_config["valid_weights"].cpu().numpy()
    test_weights = model_config["test_weights"].cpu().numpy()

    train_weights = np.unique(train_weights)
    valid_weights = np.unique(test_weights)
    test_weights = np.unique(test_weights)
    del model_config["train_weights"]
    del model_config["valid_weights"]
    del model_config["test_weights"]
    model_config["unique_train_weights"] = train_weights.tolist()
    model_config["unique_valid_weights"] = valid_weights.tolist()
    model_config["unique_test_weights"] = test_weights.tolist()

    if configurations_length < 2:
        return
    try:
        shutil.move(training_model.results_folder, exploration_path)
        config_path = os.path.join(exploration_path, training_model.date, "model_config.json")
        with open(config_path, "w") as file:
            json.dump(str(model_config), file)
    except FileNotFoundError:
        print(f"An error occurred while trying to move the result files of model {training_model.results_folder}")


def setup_exploration_folder_structure(configuration_amount, out_dir, exploration_path):
    store_path = exploration_path if exploration_path else out_dir
    if configuration_amount > 1:
        date = datetime.datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
        store_path = os.path.join(store_path, f"explorations-{date}")
        # temp_mirror_copy = os.path.join(store_path, f"explorations-{date}")
        os.mkdir(store_path)
        # os.mkdir(temp_mirror_copy)
        return store_path


def create_model_configurations(config_nr, dataset, out_dir, model_configs="exploration_config.json"):
    config_path = os.path.join(dataset.dataset_info_path, model_configs)

    with open(config_path, "r") as file_handle:
        configurations = json.load(file_handle)
    propagation_reach = configurations["propagation_reach"][config_nr]
    mlp_length = configurations["mlp_length"][config_nr]
    hidden_size = configurations["input_size"][config_nr]

    model_config = {
        "num_layers": propagation_reach,
        "hidden_size_orig": dataset.num_types,
        "gnn_h_size": hidden_size,
        "learning_rate": 0.0001,
        "batch_size": 64, # Maybe increase this size
        "num_epochs": 1,
        "num_edge_types": 5,
        "results_dir": out_dir,
        "linear_activation": nn.Sigmoid,
        "num_linear_layers": mlp_length,
    }
    return model_config


def angha_exploration(config_number):
    out_dir = os.environ.get("out_dir")
    exploration_dir = os.environ.get("exploration_dir")

    dataset = AnghabenchGraphDataset(non_empty=True)

    combinations = [
        (R.LLVMGraphBuilder, R.LLVMBPVisitor, M.GnnPytorchBranchProbabilityModel),
    ]

    model_config = create_model_configurations(config_number, dataset, out_dir)

    config_length = len(model_config)

    # files = os.listdir(out_dir)
    # create_structure = False
    # for folder in files:
    #     if "explorations" in folder:
    #         create_structure = True
    #         exploration_dir = folder
    #         break

    # if not create_structure:
    exploration_dir = setup_exploration_folder_structure(config_length, out_dir, exploration_dir)

    for builder, visitor, model in combinations:

        # clang_driver = ClangDriver(
        #     ClangDriver.ProgrammingLanguage.C,
        #     ClangDriver.OptimizationLevel.O3,
        #     [],
        #     [],
        # )

        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
        # TODO: StratifiedKFold with target being tensors with multiple y values
        #      How to calculate the distribution, since sample holds multiple ys

        train_set, valid_set, test_set = split_dataset(dataset)
        train_weights, valid_weights, test_weights = lookup_branch_weights(train_set, valid_set, test_set, dataset)

        model_config["train_weights"] = train_weights
        model_config["valid_weights"] = valid_weights
        model_config["test_weights"] = test_weights

        branch_model = model(config=model_config)
        train_summary = branch_model.train(
            train_set,
            valid_set,
        )
        test_summary = branch_model.predict_test_set(test_set)
        print(train_summary)
        print(test_summary)

        exit()
        move_results(branch_model, exploration_dir, config_length, model_config)



if __name__ == '__main__':
    config_number = 2
    # config_number = int(sys.argv[2])
    print("\n\nConfig\n\n", config_number)
    angha_exploration(config_number)
    gpu_usage()
    gc.collect()
    torch.cuda.empty_cache()
    gpu_usage()

