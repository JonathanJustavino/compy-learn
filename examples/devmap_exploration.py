import sys

import numpy as np
from absl import flags
from absl import app

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver
from compy.datasets.anghabench import AnghabenchDataset
from compy.datasets import dataflow_preprocess
from compy.datasets.dataflow_preprocess import main as dataflow_main


dataset_path = '/home/john/Documents/workspace/Studium/Masterarbeit/angha_dataset/ExtractGraphsTask/'
FLAGS = flags.FLAGS
FLAGS.outdir = dataset_path
FLAGS.preprocess = True
FLAGS.eliminate_data_duplicates = True


app.run(dataflow_main)


exit()

# Load dataset
ANGHA_FLAG = True
PREPROCESS_FLAG = not ANGHA_FLAG

dataset = D.OpenCLDevmapDataset()
if ANGHA_FLAG:
    dataset = AnghabenchDataset()
    dataset.set_content_dir(dataset_path)

#TODO einen fixen split 10% test 90% training
# funktion schreiben die die samples direkt lädt und testen

# Explore combinations
combinations = [
    # CGO 20: AST+DF, CDFG
    (R.ASTGraphBuilder, R.ASTDataVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchGeomModel),
    # Arxiv 20: ProGraML
    (R.LLVMGraphBuilder, R.LLVMProGraMLVisitor, M.GnnPytorchGeomModel),
    # PACT 17: DeepTune
    (R.SyntaxSeqBuilder, R.SyntaxTokenkindVariableVisitor, M.RnnTfModel),
    # Extra
    (R.ASTGraphBuilder, R.ASTDataCFGVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGCallVisitor, M.GnnPytorchGeomModel),

    (R.LLVMGraphBuilder, R.LLVMCDFGPlusVisitor, M.GnnPytorchGeomModel),
]

combinations = [
    #(R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchGeomModel),
    #(R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchDGLModel),
    #(R.ASTGraphBuilder, R.ASTDataVisitor, M.GnnPytorchBranchProbabilityModel),
    #(R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchBranchProbabilityModel),
    (R.LLVMGraphBuilder, R.LLVMBPVisitor, M.GnnPytorchBranchProbabilityModel),
]

for builder, visitor, model in combinations:
    print("Processing %s-%s-%s" % (builder.__name__, visitor.__name__, model.__name__))

    if PREPROCESS_FLAG:
        # Build representation
        clang_driver = ClangDriver(
            ClangDriver.ProgrammingLanguage.OpenCL,
            ClangDriver.OptimizationLevel.O3,
            [(x, ClangDriver.IncludeDirType.User) for x in dataset.additional_include_dirs],
            ["-xcl", "-target", "x86_64-pc-linux-gnu"],
        )
        data = dataset.preprocess(builder(clang_driver), visitor, ["amd-app-sdk-3.0"])
        # Train and test
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
        split = kf.split(data["samples"], [sample["info"][5] for sample in data["samples"]])

        for train_idx, test_idx in split:
            model = model(num_types=data["num_types"])
            train_summary = model.train(
                list(np.array(data["samples"])[train_idx]),
                list(np.array(data["samples"])[test_idx]),
            )
            print(train_summary)

            break
    else:
        #TODO return data in a dict with samples key and maybe "info", "x", "y", "num_types" as subkeys?
        #data = dataset.load_graphs()

        clang_driver = ClangDriver(
            ClangDriver.ProgrammingLanguage.C,
            ClangDriver.OptimizationLevel.O3,
            [],
            [],
        )

        data = dataset.load_graphs()
        # data = dataset.preprocess(builder(clang_driver), visitor)

        dataset_length = len(data['samples'])
        test_range = round(float(dataset_length) * 0.1)
        train_idx = round(dataset_length - test_range)
        train_samples = data["samples"][0:train_idx]
        test_samples = data["samples"][train_idx:dataset_length]
        #FIXME Where can the num_type be efficiently computed form?
        num_types = 43
        model = model(num_types=num_types)
        train_summary = model.train(
            train_samples,
            test_samples
        )
        print(train_summary)
