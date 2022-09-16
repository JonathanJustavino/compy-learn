import os
import tqdm
import torch
import pytest
import multiprocessing
from collections import namedtuple

from compy import models as M
from compy.datasets import AnghabenchGraphDataset
from compy.representations import RepresentationBuilder


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class TestBuilder(RepresentationBuilder):
    def string_to_info(self, src):
        functionInfo = objectview({"name": "xyz"})
        return objectview({"functionInfos": [functionInfo]})

    def info_to_representation(self, info, visitor):
        return "Repr"


@pytest.fixture
def dataset_fixture():
    ds = AnghabenchGraphDataset()
    yield ds


def test_app_dir_exists(dataset_fixture):
    assert os.path.isdir(dataset_fixture.dataset_dir)


def test_overfit(dataset_fixture):
    sample_index = 34140
    train_set = torch.utils.data.Subset(dataset_fixture, [sample_index])
    test_set = torch.utils.data.Subset(dataset_fixture, [sample_index])
    num_types = dataset_fixture.num_types

    model_config = {
        "num_layers": 16,
        "hidden_size_orig": num_types,
        "gnn_h_size": 80,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 350,
        "num_edge_types": 5,
    }

    sample = train_set.__getitem__(0)

    for branch in sample.y:
        print(branch)

    test_model = M.GnnPytorchBranchProbabilityModel(config=model_config, num_types=num_types)
    train_summary, valid_summary = test_model.train(train_set, test_set)
    train_last_epoch = train_summary[-1]
    valid_last_epoch = valid_summary[-1]

    assert abs(1.0 - train_last_epoch[0]) < 0.01
    assert abs(1.0 - train_last_epoch[1]) < 0.01
    assert abs(1.0 - train_last_epoch[2]) < 0.01
    assert abs(1.0 - valid_last_epoch[0]) < 0.01
    assert abs(1.0 - valid_last_epoch[1]) < 0.01
    assert abs(1.0 - valid_last_epoch[2]) < 0.01


def test_sample_distritbution(dataset_fixture, debug=False):
    threads = 8 if not debug else 1
    samples = multi_thread(threads=threads)
    assert len(samples) > 0
    for sample in samples:
        graph = dataset_fixture.get(sample[1])
        assert graph is not None


def _get_prob_cls(cuda=False):
    data_type = torch.float32
    PClasses = namedtuple("PClasses", ["c396", "c963", "c5050", "c3762", "c6237", "c0100", "c1000"])

    if cuda:
        return PClasses(
            torch.tensor([0.3, 0.96], dtype=data_type).cuda(),
            torch.tensor([0.96, 0.3], dtype=data_type).cuda(),
            torch.tensor([0.5, 0.5], dtype=data_type).cuda(),
            torch.tensor([0.37, 0.62], dtype=data_type).cuda(),
            torch.tensor([0.62, 0.37], dtype=data_type).cuda(),
            torch.tensor([0., 1.0], dtype=data_type).cuda(),
            torch.tensor([1.0, 0.], dtype=data_type).cuda()
        )

    return PClasses(
        torch.tensor([0.3, 0.96], dtype=data_type),
        torch.tensor([0.96, 0.3], dtype=data_type),
        torch.tensor([0.5, 0.5], dtype=data_type),
        torch.tensor([0.37, 0.62], dtype=data_type),
        torch.tensor([0.62, 0.37], dtype=data_type),
        torch.tensor([0., 1.0], dtype=data_type),
        torch.tensor([1.0, 0.], dtype=data_type)
    )


def split(num_splits, files):
    split_range, remainder = divmod(len(files), num_splits)
    return list(files[i * split_range + min(i, remainder):(i + 1) * split_range + min(i + 1, remainder)]
                for i in range(num_splits))


def multi_thread(threads=1):
    data_path, result_path = _resolve_paths()
    print(data_path, result_path)
    files = os.listdir(data_path)
    # TODO Decrease dataset by dividing by random value
    # denominator = random.randrange(0, 2000)
    chunked_files = split(threads, files)
    tasks_split_indexed = [(index, split) for index, split in enumerate(chunked_files)]
    with multiprocessing.Pool(processes=threads) as pool:
        results = list(tqdm.tqdm(pool.imap(filter_graphs, tasks_split_indexed), total=len(files), desc="Processing...", disable=False))
    pool.close()
    pool.join()
    return results


def evenly_distributed_probabilities(current_max_score=-1, graph_score=-1, file_idx=-1, graph_counts=-1):
    if current_max_score < graph_score:
        threshold = 30
        first, second, third, fourth = graph_counts
        not_null = abs(first - second < threshold) and abs(first - third < threshold) and abs(first - fourth < threshold)
        not_null = not_null and abs(second - third < threshold) and abs(second - fourth < threshold)
        not_null = not_null and abs(third - fourth < threshold)
        if not_null:
            return file_idx, graph_score
        return -1, -1


def mostly_single_class(current_max_score=-1, graph_score=-1, file_idx=-1, graph_counts=-1):
    first, second, third, fourth = graph_counts
    if first > second and first > third and current_max_score > graph_score:
        current_max_score = graph_score
        index_of_max_score = file_idx
        return index_of_max_score, current_max_score
    return -1, -1


def filter_graphs(files):
    pcls = _get_prob_cls(cuda=False)
    data_path, _ = _resolve_paths()
    index, split = files
    max_score = -1
    index_of_max_score = -1
    for file_idx, file in tqdm.tqdm(enumerate(split), total=len(split), desc=f"Processing {index}", disable=False):
        graph_counts = [0, 0, 0, 0]
        graph = _load_graph(data_path, file)
        for row in graph.y:
            graph_counts[0] += int(pcls.c396 in row)
            graph_counts[0] += int(pcls.c963 in row)
            graph_counts[1] += int(pcls.c5050 in row)
            graph_counts[2] += int(pcls.c3762 in row)
            graph_counts[2] += int(pcls.c6237 in row)
            graph_counts[3] += int(pcls.c0100 in row)
            graph_counts[3] += int(pcls.c1000 in row)

        graph_score = sum(graph_counts)
        index_of_max_score, max_score = mostly_single_class(current_max_score=max_score, graph_score=graph_score, file_idx=file_idx, graph_counts=graph_counts)
    return index_of_max_score, max_score


def _resolve_paths():
    data_path = os.environ.get('PROCESSED_DS')
    base_path = os.path.split(data_path)[0]
    return data_path, os.path.join(base_path, "dataset_info")


def _load_graph(path, name):
    file = f"{path}/{name}"
    return torch.load(file)
