import os
import re
from tqdm import tqdm
import pickle
import numpy as np
from collections import namedtuple


Ratio = namedtuple('Ratio', ['name', 'pct_03', 'pct_37', 'pct_50', 'pct_62', 'pct_96', 'pct_100'], defaults=(['file_name', 0, 0, 0, 0, 0, 0]))

dataset_path = os.environ.get('ANGHA_PICKLE_DIR')
file_name = 'dataset_ratio.pickle'
temp_name = 'temporary_ratio.pickle'
store_path = os.path.split(dataset_path)[0]
temporary_path = store_path
store_path = os.path.join(store_path, 'dataset_info', file_name)
temporary_path = os.path.join(temporary_path, 'dataset_info', temp_name)


def gather_file_probability_count(base_path, file_name):
    file_handle = os.path.join(base_path, file_name)
    probability = 'probability'
    probabilities = []

    with open(file_handle, 'rb') as f:
        data = pickle.load(f)
        samples = data["samples"]
        for sample in samples:
            edge_list = sample["x"]["code_rep"].get_edge_list_with_data()
            for element in edge_list:
                candidate = element[-1]
                if probability in candidate:
                    probabilities.append(candidate[probability])

    pct_3, pct_37, pct_50, pct_62, pct_96, pct_100 = count_probability_occurrence(probabilities)

    return file_name, pct_3, pct_37, pct_50, pct_62, pct_96, pct_100


def count_probability_occurrence(probabilities):
    probs = np.array(probabilities)
    count_prob = np.count_nonzero
    counted_probs = (
        count_prob(probs == 3),
        count_prob(probs == 37),
        count_prob(probs == 50),
        count_prob(probs == 62),
        count_prob(probs == 96),
        count_prob(probs == 100)
    )
    return counted_probs


def sort_by_number(key):
    search_for_first_number = '^.[^_]*'
    match = re.match(search_for_first_number, key)
    number = match.group(0)
    return int(number)


def store(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def process_files(store_path):
    files = os.listdir(dataset_path)
    files.sort(key=sort_by_number)
    probabilities_per_file = []

    index = 0
    for file in tqdm(files):
        if index > 20:
            break
        ratio = gather_file_probability_count(dataset_path, file)
        probabilities_per_file.append(ratio)
        index += 1

    store(probabilities_per_file, store_path)


def gather_probabilities():
    if os.path.isfile(store_path):
        process_files(temporary_path)
        should_replace = input('Replace Old File? y/N')
        if should_replace == 'y' or should_replace == 'Y':
            print("Replacing File")
            return os.replace(temporary_path, store_path)
        print(f"File has not been replaced, it is located at:\n{temporary_path}")
        return

    process_files(store_path)


def load_ratios():
    with open(store_path, 'rb') as file:
        return pickle.load(file)


def count_probabilities(probabilities_per_file):
    pct_03, pct_37, pct_50, pct_62, pct_96, pct_100 = 0, 0, 0, 0, 0, 0

    for ratio in probabilities_per_file:
        pct_03 += ratio.pct_03
        pct_37 += ratio.pct_37
        pct_50 += ratio.pct_50
        pct_62 += ratio.pct_62
        pct_96 += ratio.pct_96
        pct_100 += ratio.pct_100

    return pct_03, pct_37, pct_50, pct_62, pct_96, pct_100


WRITE = False

if WRITE:
    gather_probabilities()
else:
    ratios = load_ratios()
    probabilities_per_file = [Ratio(*item) for item in ratios]
    probs = count_probabilities(probabilities_per_file)
    probs = list(probs)
    calculate_ratio = lambda value, total: float(value) / float(total)
    total_probabilites = float(sum(probs))
    ratios = [calculate_ratio(prob, total_probabilites) for prob in probs]
    dataset_ratio = Ratio('dataset', *ratios)
    print(dataset_ratio)

