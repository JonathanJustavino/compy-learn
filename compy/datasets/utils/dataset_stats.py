import os
import re
import csv
import torch
import pickle
import threading
from tqdm import tqdm
from collections import namedtuple, Counter
from torch.utils.data import DataLoader as TorchDataLoader
from compy.datasets.anghabench_graph import AnghabenchGraphDataset


Ratio = namedtuple('Ratio', ['name', 'pct_0', 'pct_03', 'pct_37', 'pct_50', 'pct_62', 'pct_96', 'pct_100'], defaults=(['file_name', 0, 0, 0, 0, 0, 0, 0]))
LR_Ratio = namedtuple('LR_Ratio', ['name', 'left', 'right'], defaults=(['lr_ratio', 0, 0]))

dataset_path = os.environ.get('ANGHA_PICKLE_DIR')
file_name = 'dataset_ratio.pickle'
temp_name = 'temporary_ratio.pickle'
store_path = os.path.split(dataset_path)[0]
temporary_path = store_path
store_path = os.path.join(store_path, 'dataset_info', file_name)
temporary_path = os.path.join(temporary_path, 'dataset_info', temp_name)


def process_data(data):
    return [
        {
            "nodes": data["x"]["code_rep"].get_node_list(),
            "edges": data["x"]["code_rep"].get_edge_list_tuple(),
            "probability": data["x"]["code_rep"].get_edge_list_with_data(),
        }
        for data in data
    ]


def __build_pg_graphs(batch_graphs, num_types):
    pg_graphs = []
    previous_source_node = -1

    # Graph
    for graph_index, batch_graph in enumerate(batch_graphs):

        # Edges
        edge_index, edge_features, probability_list, source_nodes = [], [], [], []
        probability = "probability"
        for index, edge in enumerate(batch_graph["edges"]):
            last_element = batch_graph[probability][index][-1]
            edge_type = batch_graph[probability][index][1]
            source_node = edge[0]

            if probability in last_element and edge_type == 5:
                if source_node == previous_source_node:
                    previous_idx = len(probability_list) - 1
                    probability_list[previous_idx] = (probability_list[previous_idx], last_element[probability])
                else:
                    source_nodes.append(source_node)
                    probability_list.append(last_element[probability])
            previous_source_node = source_node

        edge_probabilities = __get_probability_tensor(probability_list)

        pg_graphs.append(edge_probabilities)

    return pg_graphs


def __get_probability_tensor(probabilities):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def remove_single_branches(branch_value): return branch_value != 100

    filtered = filter(remove_single_branches, probabilities)
    edge_probabilities = list(filtered)

    return torch.tensor(edge_probabilities, dtype=torch.float, device=device) / 100


def count_percentages(graph):
    pct_0, pct_3, pct_37, pct_50, pct_62, pct_96, pct_100 = 0, 0, 0, 0, 0, 0, 0
    def count_percentage(graph_tensor, percentage): return torch.where(graph_tensor == percentage, 1, 0).count_nonzero().item()

    pct_0 += count_percentage(graph, 0.0)
    pct_3 += count_percentage(graph, 0.03)
    pct_37 += count_percentage(graph, 0.37)
    pct_50 += count_percentage(graph, 0.50)
    pct_62 += count_percentage(graph, 0.62)
    pct_96 += count_percentage(graph, 0.96)
    pct_100 += count_percentage(graph, 1.0)

    return [pct_0, pct_3, pct_37, pct_50, pct_62, pct_96, pct_100]


def left_right_ratio(graph):
    def l_ratio(graph_tensor): return torch.where(graph_tensor > 0.5, 1, 0).count_nonzero().item()
    def r_ratio(graph_tensor): return torch.where(graph_tensor > 0.5, 1, 0).count_nonzero().item()
    lhs = graph[:, 0]
    rhs = graph[:, 1]
    return [l_ratio(lhs), r_ratio(rhs)]


def store(data, path):
    with open(path, 'wb') as file:
        for item in data:
            pickle.dump(item, file)


def process_batches(dataset, start_idx=0):
    dataset_path = os.environ.get('ANGHA_PICKLE_DIR')
    store_path = os.path.split(dataset_path)[0]
    store_path = os.path.join(store_path, "dataset_info", "processed_batches")

    batch_size = 64
    torch_loader = TorchDataLoader(dataset=dataset, batch_size=batch_size, collate_fn=process_data)

    for batch in tqdm(torch_loader):
        graphs = __build_pg_graphs(batch, dataset.num_types)
        file_name = f"batch-{start_idx}.pickle"
        save_file = os.path.join(store_path, file_name)
        with open(save_file, 'wb') as file:
            pickle.dump(graphs, file)
        start_idx += 1
    print("Done")


def load_batches():
    basepath = os.path.split(store_path)[0]
    processed_dir = "processed_batches"
    path = os.path.join(basepath, processed_dir)
    files = [f"{path}/{file}" for file in os.listdir(path)]

    percentages = [0, 0, 0, 0, 0, 0, 0]
    lr_ratio = [0, 0]

    for file in tqdm(files):
        with open(file, 'rb') as file_handle:
            batch = pickle.load(file_handle)
            for graph in batch:
                if len(graph) < 1:
                    continue
                batch_percentages = count_percentages(graph)
                lr_ratio_batch = left_right_ratio(graph)
                percentages = list(map(lambda total_percentag_count, batch_percentage_count: total_percentag_count + batch_percentage_count, percentages, batch_percentages))
                lr_ratio = list(map(lambda total_lr_count, batch_lr_count: total_lr_count + batch_lr_count, lr_ratio, lr_ratio_batch))

    file_name = "class_ratio_left_right_ratio.pickle"
    data = {
        "lr_ratio": LR_Ratio('LR_Ratio', *lr_ratio),
        "class_ratio": Ratio('dataset', *percentages),
    }

    path = os.path.join(basepath, file_name)
    store(data, path)

    print("Done.")


def load_info():
    basepath = os.path.split(store_path)[0]
    file = f"{basepath}/class_ratio_left_right_ratio.pickle"
    with open(file, 'rb') as file_handle:
        data = pickle.load(file_handle)
        lr_count, cls_count = data.values()
        lr_ratio, cls_ratio = calculate_ratio(lr_count, cls_count)
        print("LR Ratio", lr_ratio)
        print("Class Ratio", cls_ratio)


def calculate_ratio(lr_ratio, class_ratio):
    def lr(lr_count):
        left = float(lr_count.left)
        right = float(lr_count.right)
        total = float(sum(lr_count[1:]))
        return left / total, right / total

    def cls_ratio(cls_count):
        total = sum(cls_count[1:])
        ratios = [float(item) / float(total) for item in cls_count[1:]]
        return tuple(ratios)

    return lr(lr_ratio), cls_ratio(class_ratio)


def threaded():
    # FIXME: Buggy  - eats to much RAM
    dataset = AnghabenchGraphDataset()
    split_value = 2
    split = round(dataset.total_num_samples / split_value)

    range_1 = [i for i in range(0, split)]
    range_2 = [i for i in range(split, split * 2)]

    subset_1 = torch.utils.data.Subset(dataset, range_1)
    subset_2 = torch.utils.data.Subset(dataset, range_2)

    t1 = threading.Thread(target=process_batches, args=subset_1, kwargs={'thread_name': 'first'})
    t2 = threading.Thread(target=process_batches, args=subset_2, kwargs={'thread_name': 'second', 'start_idx': range_2[0]})

    t1.start()
    t2.start()


def generate_csv():
    basepath = os.path.split(store_path)[0]
    processed_dir = "processed_batches"
    path = os.path.join(basepath, processed_dir)
    files = [f"{path}/{file}" for file in os.listdir(path)]
    csv_file = os.path.join(basepath, "dataset_ratio.csv")
    index = 0
    header = ["batch_file", "0", "3", "37", "50", "62", "96", "100", "left", "right"]

    if os.path.isfile(csv_file):
        response = input("File already exists. Do you really want to overwrite it? y/N\n")
        if response != "y" or response != "Y":
            return

    with open(csv_file, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        for file_name in tqdm(files):
            with open(file_name, 'rb') as file:
                graphs = pickle.load(file)
                for graph in graphs:
                    if len(graph) < 1:
                        continue
                    percent_count = count_percentages(graph)
                    lr_ratio = left_right_ratio(graph)
                    item = f"batch_{index}"
                    writer.writerow([item] + percent_count + lr_ratio)
                    index += 1


def compute_num_type_stats():
    max_num_types = 0
    store_data = {}
    pickle_files = os.listdir(dataset_path)
    sorting_pattern = "(\d+)"
    re_sorting = re.compile(sorting_pattern)
    numerical_sort = lambda file_name: int(re_sorting.match(file_name).group())
    pickle_files.sort(key=numerical_sort)
    num_types = []
    for file in tqdm(pickle_files, desc="Computing max num_types of graphs..."):
        filename = f"{dataset_path}/{file}"
        with open(filename, "rb") as f:
            collection = pickle.load(f)
            num_type = collection["num_types"]
            if num_type > max_num_types:
                max_num_types = num_type
            num_types.append(num_type)

    distribution = Counter(num_types)
    store_data["num_types_per_file"] = num_types
    store_data["distribution"] = distribution
    store_data["max_num_type"] = max_num_types
    store_data["most_common"] = distribution.most_common()

    basepath = os.path.split(store_path)[0]
    filename = os.path.join(basepath, "num_types_distribution.pickle")

    with open(filename, "rb") as f:
        pickle.dump(store_data, f)


def split_dataset(dataset, subsample=0):
    amount_samples = dataset.total_num_samples
    if subsample:
        amount_samples = amount_samples // subsample
    test_range = round(float(amount_samples) * 0.1)
    valid_range = round(float(amount_samples) * 0.1)
    train_range = round(amount_samples - (test_range + valid_range))

    train_samples = [index for index in range(0, train_range)]
    validation_samples = [index for index in range(train_range, (train_range + valid_range))]
    test_samples = [index for index in range((train_range + valid_range), amount_samples)]

    training_set = torch.utils.data.Subset(dataset, train_samples)
    validation_set = torch.utils.data.Subset(dataset, validation_samples)
    test_set = torch.utils.data.Subset(dataset, test_samples)

    return training_set, validation_set, test_set
