import os
import sys
import numpy as np
import torch.utils.data
import tqdm
from absl import flags
from absl import app

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver
from compy.datasets.anghabench_graph import AnghabenchGraphDataset
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
print(dataset)

combinations = [
    (R.LLVMGraphBuilder, R.LLVMBPVisitor, M.GnnPytorchBranchProbabilityModel),
]


def compute_branches_per_set(training_subset, test_subset, dataset):
    total_test_branches = 0
    total_train_branches = 0
    all_branches = dataset.total_branches

    training_sample_count = len(training_subset)
    test_sample_count = len(test_subset)
    print("Computing amount of train and test branches")
    if training_sample_count + test_sample_count == len(dataset):
        for sample in tqdm.tqdm(test_subset):
            total_test_branches += len(sample.y)

        total_train_branches = all_branches - total_test_branches
        setattr(training_subset, 'total_branches', total_train_branches)
        setattr(test_subset, 'total_branches', total_test_branches)

        print(f"Amount of training branches {total_train_branches}"
              f"Amount of test branches {total_test_branches}")
        return

    for sample in tqdm.tqdm(test_subset):
        total_test_branches += len(sample.y)

    for sample in tqdm.tqdm(training_subset):
        total_train_branches += len(sample.y)

    setattr(training_subset, 'total_branches', total_train_branches)
    setattr(test_subset, 'total_branches', total_test_branches)

    print(f"Amount of training branches {total_train_branches}"
          f"Amount of test branches {total_test_branches}")


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
    amount_samples = amount_samples // 1000
    test_range = round(float(amount_samples) * 0.1)
    train_range = round(amount_samples - test_range)

    train_samples = [index for index in range(0, train_range)]
    test_samples = [index for index in range(train_range, amount_samples)]

    # high_branch_count_samples 53541, 124106, 20367, 92786, 60315, 118622
    # similar_distributed_samples = [(106, 148403), (71, 111665), (65, 16409), (74, 157018), (65, 11193), (77, 118622)]
    # unneeded_sample_score = 0
    # difficult_sample = [(unneeded_sample_score, 34140)]
    # idx_0100 = 52030
    # two_balanced = [961679, idx_0100]
    # balanced = [961679, 33631, 929414, idx_0100]

    # train_samples = balanced
    # test_samples = balanced

    train_set = torch.utils.data.Subset(dataset, train_samples)
    test_set = torch.utils.data.Subset(dataset, test_samples)

    compute_branches_per_set(train_set, test_set, dataset)


    num_types = dataset.num_types
    model_config = {
        "num_layers": 16,
        "hidden_size_orig": num_types,
        "gnn_h_size": 80,
        "learning_rate": 0.001,
        "batch_size": 32, # Maybe increase this size
        "num_epochs": 600,
        "num_edge_types": 5,
    }

    branch_model = model(config=model_config, num_types=num_types)
    train_summary = branch_model.train(
        train_set,
        test_set
    )
    print(train_summary)
