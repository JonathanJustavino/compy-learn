import os
import pickle
import sys
import numpy as np
import torch.utils.data
import tqdm
from absl import flags
from absl import app
from collections import  namedtuple

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver
from compy.datasets.anghabench_graph import AnghabenchGraphDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from compy.datasets import dataflow_preprocess
from compy.datasets.dataflow_preprocess import main as dataflow_main


# dataset_path = '/net/home/luederitz/anghabench'
# FLAGS = flags.FLAGS
# FLAGS.out_dir = dataset_path
# FLAGS.preprocess = True
# FLAGS.eliminate_data_duplicates = True
# FLAGS.debug = False


# Load dataset
ANGHA_FLAG = True
PREPROCESS_FLAG = not ANGHA_FLAG

dataset = AnghabenchGraphDataset(non_empty=True)

combinations = [
    (R.LLVMGraphBuilder, R.LLVMBPVisitor, M.GnnPytorchBranchProbabilityModel),
]


def recompute_branch_weights(training_subset, test_subset, batch_size=512, num_workers=3):
    max_branch_length = 101
    training_weights = torch.zeros(max_branch_length).cuda()
    testing_weights = torch.zeros(max_branch_length).cuda()

    train_loader = GeometricDataLoader(training_subset, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_subset, batch_size=batch_size, num_workers=num_workers)

    if not torch.cuda.is_available():
        print("GPU not available")
        exit()

    print("Computing amount of train and test branches")
    train_branch_count = 0
    for sample in tqdm.tqdm(train_loader):
        sample = sample.cuda()
        target = sample.y * 100
        target = target.int()
        train_bincount = torch.bincount(target.view(-1), minlength=max_branch_length).int()
        training_weights += train_bincount
        train_branch_count += len(sample.y)

    test_branch_count = 0
    for sample in tqdm.tqdm(test_loader):
        sample = sample.cuda()
        target = sample.y * 100
        target = target.int()
        test_bincount = torch.bincount(target.view(-1), minlength=max_branch_length).int()
        testing_weights += test_bincount
        test_branch_count += len(sample.y)

    total_train_probabilities = sum(training_weights).int().item()
    total_test_probabilities = sum(testing_weights).int().item()

    training_weights = training_weights.div(total_train_probabilities)
    testing_weights = testing_weights.div(total_test_probabilities)
    classes = [0, 3, 37, 50, 62, 96, 100]
    class_count = len(classes)

    for index in classes:
        training_weights[index] = 1 / class_count
        testing_weights[index] = 1 / class_count

    total_train_branches = round(total_train_probabilities / 2)
    total_test_branches = round(total_test_probabilities / 2)

    return training_weights, testing_weights, total_train_branches, total_test_branches


def lookup_branch_weights(training_subset, test_subset, dataset, cache_path="cache/cached_weights.pt", recompute=False):
    cache_path = f"{dataset.dataset_info_path}/{cache_path}"
    cache_exists = os.path.exists(cache_path)
    train_indices = []
    test_indices = []

    if cache_exists and not recompute:
        print("Using cached weights")
        cached_weights = torch.load(cache_path)
        train_weights = cached_weights["train_weights"]
        test_weights = cached_weights["test_weights"]
        train_indices = cached_weights["train_indices"]
        test_indices = cached_weights["test_indices"]
        total_train_branches = cached_weights["total_train_branches"]
        total_test_branches = cached_weights["total_test_branches"]

    if recompute or not (train_indices == training_subset.indices and test_indices == test_subset.indices):
        train_weights, test_weights, total_train_branches, total_test_branches = recompute_branch_weights(training_subset, test_subset, batch_size=512, num_workers=3)

        print("Caching results")
        new_weights = {
            'train_weights': train_weights,
            'test_weights': test_weights,
            'train_indices': training_subset.indices,
            'test_indices': test_subset.indices,
            'total_train_branches': total_train_branches,
            'total_test_branches': total_test_branches
        }

        torch.save(new_weights, cache_path)

    setattr(training_subset, 'total_branches', total_train_branches)
    setattr(test_subset, 'total_branches', total_test_branches)

    print(f"Amount of training branches {total_train_branches}\n"
          f"Amount of test branches {total_test_branches}")

    return train_weights, test_weights


for builder, visitor, model in combinations:

    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O3,
        [],
        [],
    )

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    print(kf)
    # TODO: StratifiedKFold with target being tensors with multiple y values
    #      How to calculate the distribution, since sample holds multiple ys

    amount_samples = dataset.total_num_samples
    amount_samples = amount_samples
    test_range = round(float(amount_samples) * 0.1)
    valid_range = round(float(amount_samples) * 0.1)
    train_range = round(amount_samples - (test_range + valid_range))

    # high_branch_count_samples 53541, 124106, 20367, 92786, 60315, 118622
    # similar_distributed_samples = [(106, 148403), (71, 111665), (65, 16409), (74, 157018), (65, 11193), (77, 118622)]
    # unneeded_sample_score = 0
    # difficult_sample = [(unneeded_sample_score, 34140)]
    # idx_0100 = 52030
    # two_balanced = [961679, idx_0100]
    # balanced = [961679, 33631, 929414, idx_0100]

    # train_samples = balanced
    # test_samples = balanced

    # import random
    # non_empty_samples = [i for i in range(len(dataset.non_empty_samples))]
    # amount = 500
    # samples = random.sample(non_empty_samples, k=amount)

    train_samples = [index for index in range(0, train_range)]
    validation_samples = [index for index in range(0, valid_range)]
    test_samples = [index for index in range(train_range, amount_samples)]

    # train_samples = samples
    # test_samples = samples

    train_set = torch.utils.data.Subset(dataset, train_samples)
    valid_set = torch.utils.data.Subset(dataset, validation_samples)

    train_weights, test_weights = lookup_branch_weights(train_set, valid_set, dataset)

    num_types = dataset.num_types
    model_config = {
        "num_layers": 8,
        "hidden_size_orig": num_types,
        "gnn_h_size": 80,
        "learning_rate": 0.0001,
        "batch_size": 128, # Maybe increase this size
        "num_epochs": 1100,
        "num_edge_types": 5,
    }

    branch_model = model(config=model_config, num_types=num_types, train_weights=train_weights, test_weights=test_weights)
    train_summary = branch_model.train(
        train_set,
        valid_set,
    )
    print(train_summary)
