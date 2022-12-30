import sys
import numpy as np
import torch.utils.data
from absl import flags
from absl import app

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver
from compy.datasets.llvm_dataset import LLVMDataset
from compy.datasets import dataflow_preprocess
from compy.datasets.dataflow_preprocess import main as dataflow_main


dataset_path = '/home/john/Documents/workspace/Studium/Masterarbeit/llvm_dataset'
FLAGS = flags.FLAGS
FLAGS.out_dir = dataset_path
FLAGS.preprocess = True
FLAGS.eliminate_data_duplicates = True
FLAGS.debug = False


# Load dataset

# ANGHA_FLAG = True
# PREPROCESS_FLAG = not ANGHA_FLAG
#
# dataset = D.OpenCLDevmapDataset()
# if ANGHA_FLAG:
#     dataset = AnghabenchGraphDataset()

# Explore combinations
combinations = [
    (R.LLVMGraphBuilder, R.LLVMBPVisitor, M.GnnPytorchBranchProbabilityModel),
]

for builder, visitor, model in combinations:
    print("Processing %s-%s-%s" % (builder.__name__, visitor.__name__, model.__name__))

    # if PREPROCESS_FLAG:
    #     # Build representation
    #     clang_driver = ClangDriver(
    #         ClangDriver.ProgrammingLanguage.OpenCL,
    #         ClangDriver.OptimizationLevel.O3,
    #         [(x, ClangDriver.IncludeDirType.User) for x in dataset.additional_include_dirs],
    #         ["-xcl", "-target", "x86_64-pc-linux-gnu"],
    #     )
    #     data = dataset.preprocess(builder(clang_driver), visitor, ["amd-app-sdk-3.0"])
    #     # Train and test
    #     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    #     split = kf.split(data["samples"], [sample["info"][5] for sample in data["samples"]])

    # for train_idx, test_idx in split:
    #     model = model(num_types=data["num_types"])
    #     train_summary = model.train(
    #         list(np.array(data["samples"])[train_idx]),
    #         list(np.array(data["samples"])[test_idx]),
    #     )
    #     print(train_summary)
    #
    #     break

    prof_path = "/home/john/Documents/workspace/Studium/Masterarbeit/profdata/clang.profdata"
    prof_flag = "-fprofile-use"
    prof_cmd = f"{prof_flag}={prof_path}"
    comp_path = "/home/john/.local/share/compy-Learn/1.0/LLVMDataset/content/stage2-prof-gen"
    comp_cmd = f"-p {comp_path}"
    compiler_flags = [comp_cmd]
    include_dirs = []

    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O0,
        include_dirs,
        compiler_flags
        # ["-xcl", "-target", "x86_64-pc-linux-gnu"],
    )

    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    dataset = LLVMDataset()
    # app.run(dataflow_main)
    data = dataset.preprocess(builder(clang_driver), visitor)
    print(data)


    # amount_samples = dataset.total_num_samples
    # amount_samples = amount_samples // 1000
    # test_range = round(float(amount_samples) * 0.1)
    # train_range = round(amount_samples - test_range)
    # # train_samples = dataset.graph_indexes[0:train_range]
    # # test_samples = dataset.graph_indexes[train_range:amount_samples]
    #
    # train_samples = [index for index in range(0, train_range)]
    # test_samples = [index for index in range(train_range, amount_samples)]
    #
    # # train 893213
    # # test 99246
    #
    # #train_set, test_set = torch.utils.data.random_split(dataset, [train_range, test_range])
    # train_set = torch.utils.data.Subset(dataset, train_samples)
    # test_set = torch.utils.data.Subset(dataset, test_samples)
    #
    # model = model(num_types=dataset.num_types)
    # train_summary = model.train(
    #     train_set,
    #     test_set
    # )
    # print(train_summary)
