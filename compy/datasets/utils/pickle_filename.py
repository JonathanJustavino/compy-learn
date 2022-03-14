import re


PICKLEFILE_REGEX = re.compile(r'(\d+)_samples(\d+)\.pickle')


def get_pickle_filename(file_index, num_samples):
    return f"{file_index}_samples{num_samples}.pickle"


def parse_pickle_filename(file_name):
    result = re.match(PICKLEFILE_REGEX, file_name)
    file_index = int(result.group(1))
    num_samples = int(result.group(2))
    return file_index, num_samples


