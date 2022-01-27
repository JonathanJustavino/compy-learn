import copy
import os
import multiprocessing
import networkx as nx
import pickle
import sys
import uuid
import fnmatch

import tqdm
from absl import app
from absl import flags

from compy import datasets
from compy.datasets import anghabench
import compy.representations as R
from compy.representations.extractors import ClangDriver
# from compy.representations.dataflow import DFGraphBuilder
# from compy.representations.dataflow.dominators import compute_dominators
# from compy.representations.dataflow.liveness import compute_liveness
# from compy.representations.dataflow.strong_liveness import compute_strong_liveness
# from compy.representations.dataflow.possibly_undefined import PossiblyUndefined
# from compy.representations import LLVMCFGVisitor
# from compy.representations import LLVMCFGFixPHIVisitor
# from dataflow_utils import *


sys.setrecursionlimit(100000)


# Modes
flags.DEFINE_bool('preprocess', True, 'Do preprocessing for DFA analyses.')
flags.DEFINE_bool('stats', False, 'Report stats.')
flags.DEFINE_bool('dot', False, 'Write as dot graphs.')

# Duplicate elimination criteria
flags.DEFINE_bool('eliminate_control_duplicates', False, 'Eliminate duplicates based on control properties only.')
flags.DEFINE_bool('eliminate_data_duplicates', False, 'Eliminate duplicates based on control and data properties.')


dataset_path = '/home/john/Documents/workspace/Studium/Masterarbeit/angha_dataset/ExtractGraphsTask/'

flags.DEFINE_string('out_dir', '', 'Root dir to store the results.')
flags.DEFINE_bool('debug', False, 'Single-process mode for debugging.')
FLAGS = flags.FLAGS


class MultiProcessedTask(object):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose):
        self.base_dir = base_dir
        self.num_processes = num_processes
        self.num_tasks = num_tasks
        self.previous_task = previous_task
        self.verbose = verbose

        self.in_dir = self.previous_task.out_dir if self.previous_task else None
        self.out_dir = os.path.join(self.base_dir, self.__get_tasks_str())

    def run(self):
        # Load all tasks and split them up
        tasks = self._load_tasks()
        tasks_split = self.__split(tasks, self.num_tasks)
        tasks_split_indexed = [(i, s) for i, s in enumerate(tasks_split)]

        # Run the splits
        if FLAGS.debug:
            # - In the same process
            list(map(self._process_tasks, tasks_split_indexed))
        else:
            # - In a multiprocessing pool
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                list(tqdm.tqdm(pool.imap(self._process_tasks,
                    tasks_split_indexed), total=len(tasks_split_indexed),
                    desc="Processing %s" % self.__class__.__name__))

    def _process_tasks(self, indexed_tasks: list):
        tasks_idx, tasks = indexed_tasks
        for task_idx, task in tqdm.tqdm(enumerate(tasks),
                                        desc="Processing %s %d" % (self.__class__.__name__, tasks_idx),
                                        disable=not self.verbose):
            self._process_task(task, tasks_idx)

    def _load_tasks(self):
        raise NotImplemented

    def _process_task(self, tasks: list, tasks_idx: str):
        raise NotImplemented

    def __split(self, a, n):
        k, m = divmod(len(a), n)
        return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def __get_tasks_str(self):
        tasks = [self.__class__.__name__]

        it = self.previous_task
        while it:
            tasks.append(it.__class__.__name__)
            it = it.previous_task
        tasks.reverse()

        return '_'.join(tasks)

    def __repr__(self):
        return self.__class__.__name__ + ": " + str(self.__dict__)


class ExtractGraphsTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task=None, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)
        self.dataset = anghabench.AnghabenchDataset()

    def _load_tasks(self):
        num_samples = self.dataset.get_size()
        tasks = list(range(num_samples))

        return tasks

    def _process_tasks(self, indexed_tasks):
        task_idx, tasks = indexed_tasks

        # If debug, print output directly to console. If not, write to log files
        if not FLAGS.debug:
            self.__enable_file_logging(task_idx)
        data = self.__extract_graphs(tasks[0], tasks[-1] - tasks[0])
        samples = self.__compute_samples(data)

        store(samples, os.path.join(self.out_dir, "%d.pickle" % task_idx))

    def __enable_file_logging(self, idx):
        log_dir = os.path.join(FLAGS.out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        sys.stdout = open(os.path.join(log_dir, "%d.stdout" % idx), "a")
        sys.stderr = open(os.path.join(log_dir, "%d.stderr" % idx), "a")

    def __extract_graphs(self, start_at, num_samples):
        clang_driver = ClangDriver(
            ClangDriver.ProgrammingLanguage.C,
            ClangDriver.OptimizationLevel.O0,
            [],
            []
        )

        data = self.dataset.preprocess(R.LLVMGraphBuilder(clang_driver),
                                  R.LLVMBPVisitor,
                                  start_at=start_at,
                                  num_samples=num_samples)

        return data

    def __compute_samples(self, data):
        samples = []

        for d in tqdm.tqdm(data['samples'], desc="Building analysis infos"):
            code_rep = d['x']['code_rep']
            name = d['info']

            if len(code_rep.get_node_list()) == 0:
                continue

            code_rep.relabel_nodes_to_ints()

            #FIXME debug wether d['info'] holds more information than name
            # in order to be able to shape the angha dataset like the opencldevmap
            # todo repo doppeln für paralleles debugging
            samples.append({
                "x": {"code_rep": code_rep},
                "info": name})

        return samples


class SortByGraphSizeTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames = get_pickle_files(self.in_dir)

        return filenames

    def _process_tasks(self, indexed_tasks):
        task_idx, filenames = indexed_tasks

        buckets = {}
        for filename in tqdm.tqdm(filenames, desc="Processing %s %d" % (self.__class__.__name__, task_idx)):
            with open(filename, 'rb') as f:
                samples = pickle.load(f)

            for sample in samples:
                graph_size = len(sample["code_rep"].G)

                if graph_size not in buckets:
                    buckets[graph_size] = []

                buckets[graph_size].append(sample)

        self.__store_buckets(buckets, str(task_idx))

    def __store_buckets(self, buckets, suffix=''):
        for bucket_size, bucket in buckets.items():
            store(bucket, os.path.join(self.out_dir, "%d_%s.pickle" % (bucket_size, suffix)))


class ReduceTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames_all = get_files(self.in_dir, suffix='.pickle')
        graph_sizes = get_sorted_graph_sizes(filenames_all)

        return graph_sizes

    def _process_tasks(self, indexed_tasks: list):
        if self.num_tasks == 1:
            filenames = get_files(self.in_dir, suffix='.pickle')
            bucket = self.__filenames_to_bucket(filenames)

            store(bucket, os.path.join(self.out_dir, "all.pickle"))
        else:
            super()._process_tasks(indexed_tasks)

    def _process_task(self, graph_size, tasks_idx):
        filenames = get_files(self.in_dir, prefix='%d_' % graph_size, suffix='.pickle')
        bucket = self.__filenames_to_bucket(filenames)

        store(bucket, os.path.join(self.out_dir, "%s.pickle" % graph_size))

    def __filenames_to_bucket(self, filenames):
        bucket_full = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                try:
                    bucket_part = pickle.load(f)
                except (TypeError, EOFError) as e:
                    print(e)
                    continue

                if type(bucket_part) is list:
                    bucket_full += bucket_part
                else:
                    bucket_full += [bucket_part]
        return bucket_full


class EliminateDuplicatesTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames = get_pickle_files(self.in_dir)

        return filenames

    def _process_task(self, filename, tasks_idx):
        with open(filename, 'rb') as f:
            bucket = pickle.load(f)

        seen = {}
        for sample in bucket:
            sample['code_rep'].remove_parallel_identical_edges()

            # FIXME: Use better criteria for graph equality
            code_rep = self._to_str(sample['code_rep'].G)

            # Check if already seen
            if code_rep not in seen:
                sample["name"] = [sample["name"]]
                sample["id"] = str(uuid.uuid4())
                seen[code_rep] = sample
            else:
                seen[code_rep]["name"].append(sample["name"])

        bucket_unique = []
        for code_rep, sample in seen.items():
            bucket_unique.append(sample)

        store(bucket_unique, os.path.join(self.out_dir, os.path.basename(filename)))

    def _to_str(self, code_rep):
        raise NotImplementedError


class EliminateControlDuplicatesTask(EliminateDuplicatesTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _to_str(self, code_rep):
        code_rep_str = (
            str([n.id for n in code_rep.nodes()]),
            str([(x.id, y.id) for x, y, _ in list(code_rep.edges)])
        )
        return code_rep_str


class EliminateDataDuplicatesTask(EliminateDuplicatesTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _to_str(self, code_rep):
        code_rep_str = (
            str([n.id for n in code_rep.nodes()]),
            str([(n.id, [str(d) for d in n.defs], [str(u) for u in n.uses]) for n in code_rep.nodes()]),
            str([(x.id, y.id) for x, y, _ in list(code_rep.edges)])
        )
        return code_rep_str


class AnalysisTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames = get_pickle_files(self.in_dir)

        return filenames

    def _process_task(self, filename, tasks_idx):
        try:
            with open(filename, 'rb') as f:
                bucket = pickle.load(f)
        except (EOFError, KeyError) as e:
            print(e)
            return

        split_idx = 0
        samples_augmented = []
        for sample_idx, sample in tqdm.tqdm(enumerate(bucket), desc="- Computing dataflow analysis in %s" % os.path.basename(filename), disable=not self.verbose):
            # Fact infos
            n_vars = len(sample["code_rep"].get_vars())

            # Code representation
            code_rep = sample["code_rep"]
            instrs_to_ints = self._get_coderep_instrs_to_ints_mapping(code_rep)
            code_rep = self._remap_coderep_instrs_to_ints(code_rep, instrs_to_ints)

            # Dataflow analyses
            dominators, dominators_ffs = self._compute_analysis(sample["code_rep"], compute_dominators)
            liveness, liveness_ffs = self._compute_analysis(sample["code_rep"], compute_liveness)
            s_liveness, s_liveness_ffs = self._compute_analysis(sample["code_rep"], compute_strong_liveness)

            sample_augmented = {
                "id": sample["id"],
                "name": sample["name"],
                "code_rep": code_rep,
                "n_vars": n_vars,
                "analyses": {
                    "dominators": {
                        "ffs": dominators_ffs,
                        "rounds_done": dominators.rounds_done,
                        "rounds_max": dominators.rounds_max,
                        "trace": self._flatten_trace(self._remap_trace_instrs_to_ints(dominators.trace, instrs_to_ints))
                    },
                    "liveness": {
                        "ffs": liveness_ffs,
                        "rounds_done": liveness.rounds_done,
                        "rounds_max": liveness.rounds_max,
                        "trace": self._flatten_trace(self._remap_trace_instrs_to_ints(liveness.trace, instrs_to_ints))
                    },
                    "strong_liveness": {
                        "ffs": s_liveness_ffs,
                        "rounds_done": s_liveness.rounds_done,
                        "rounds_max": s_liveness.rounds_max,
                        "trace": self._flatten_trace(self._remap_trace_instrs_to_ints(s_liveness.trace, instrs_to_ints))
                    }
                }
            }
            samples_augmented.append(sample_augmented)

            split_size = 1
            if sample_idx % split_size == 0 or sample_idx == len(bucket) - 1:
                filen = os.path.basename(filename).split('.')[0] + '_' + str(split_idx) + '.pickle'
                store(samples_augmented, os.path.join(self.out_dir, filen))
                split_idx += 1
                samples_augmented = []

            samples_max = None
            if samples_max and sample_idx > samples_max:
                break

    def _compute_analysis(self, code_rep, analysis_fn):
        # Analysis
        df, df_values = analysis_fn(code_rep)

        # Get out set as sorted list (by the nodes ids)
        df_out = list(df_values.items())
        df_out = sorted(df_out, key=lambda k: k[0].id)

        df_out3 = {}
        for k, vars in df_out:
            vars_new = set()
            for var in vars:
                if type(var) is int:
                    vars_new.add(var)
                else:
                    vars_new.add(var.id)
            df_out3[k.id] = vars_new

        return df, df_out3

    def _get_coderep_instrs_to_ints_mapping(self, code_rep):
        mapping = {}
        for node, data in code_rep.G.nodes(data=True):
            mapping[node] = data['id']
        return mapping

    def _remap_coderep_instrs_to_ints(self, code_rep, mapping):
        code_rep = copy.copy(code_rep)
        code_rep.G = nx.relabel_nodes(code_rep.G, mapping)

        return code_rep

    def _remap_trace_instrs_to_ints(self, trace, mapping):
        remapped_trace = []
        for step in trace:
            node, in_edges = step[0], step[1]

            remapped_node = mapping[node]

            remapped_in_edges = []
            for in_edge in in_edges:
                src, dst = in_edge[0], in_edge[1]
                remapped_in_edges.append((mapping[src], mapping[dst]))

            remapped_trace.append((remapped_node, remapped_in_edges))
        return remapped_trace

    def _flatten_trace(self, trace):
        flat_trace = []
        for step in trace:
            node, in_edges = step[0], step[1]

            if len(in_edges):
                for in_edge in in_edges:
                    flat_trace.append((node, [in_edge]))
            else:
                flat_trace.append((node, []))

        return flat_trace


class ComputeStatsTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames = get_pickle_files(self.in_dir)
        graph_sizes = get_sorted_graph_sizes(filenames)

        return graph_sizes

    def _process_task(self, graph_size, tasks_idx):
        nums_uniques_by_size = {}
        dfa_rounds_by_size = {}

        filenames_current_size = get_files(self.in_dir, prefix='%d_' % graph_size, suffix='.pickle')

        nums_uniques_by_size[graph_size] = []
        dfa_rounds_by_size[graph_size] = []
        for filename in filenames_current_size:
            with open(filename, 'rb') as f:
                try:
                    bucket = pickle.load(f)
                except EOFError as e:
                    print(e)
                    continue

                for sample in bucket:
                    # Num uniques
                    nums_uniques_by_size[graph_size].append(len(sample['name']))
                    # DFA rounds
                    dfa_rounds_by_size[graph_size].append({
                        'rounds_done': sample['rounds_done'],
                        'rounds_max': sample['rounds_max']})

        # Num unique
        nums_unique_by_size = {}
        for k, v in nums_uniques_by_size.items():
            nums_unique_by_size[k] = len(v)

        stats = {'nums_unique_by_size': nums_unique_by_size,
                 'nums_uniques_by_size': nums_uniques_by_size,
                 'dfa_rounds_by_size': dfa_rounds_by_size}
        store(stats, os.path.join(self.out_dir, "%s_%s.pickle" % (graph_size, tasks_idx)))


class PrintStatsTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        return [os.path.join(self.in_dir, 'all.pickle')]

    def _process_task(self, filename, tasks_idx):
        with open(filename, 'rb') as f:
            bucket = pickle.load(f)
        print('\nStats:')
        print(bucket)


class DotGraphTask(MultiProcessedTask):
    def __init__(self, base_dir, num_processes, num_tasks, previous_task, verbose=False):
        super().__init__(base_dir, num_processes, num_tasks, previous_task, verbose)

    def _load_tasks(self):
        filenames = get_pickle_files(self.in_dir)

        return filenames

    def _process_task(self, filename, tasks_idx):
        graph_size = os.path.basename(filename).split('.')[0]
        subdir = os.path.join(self.out_dir, graph_size)
        os.makedirs(subdir, exist_ok=True)

        with open(filename, 'rb') as f:
            samples = pickle.load(f)
            for sample in tqdm.tqdm(samples, desc='Writing dot graphs of %s' % filename):
                subdir_and_filename = os.path.join(subdir, sample["id"])
                sample['code_rep'].draw(subdir_and_filename + '.png')

                with open(subdir_and_filename + '.dot', 'wb') as f:
                    f.write(sample['code_rep'].draw())


def run_tasks(tasks):
    print('All tasks')
    for i, task in enumerate(tasks):
        print(i, task)
        print()

    for task in tasks:
        print('Running task', task)
        task.run()


def main(argv):
    print("Debugging status:", FLAGS.debug)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    num_cpus = multiprocessing.cpu_count()

    # Define tasks
    # - Common
    c1 = ExtractGraphsTask(
            base_dir=FLAGS.out_dir,
            num_processes=num_cpus,
            num_tasks=50000)
    c2 = SortByGraphSizeTask(
            base_dir=FLAGS.out_dir,
            num_processes=num_cpus,
            num_tasks=200,
            previous_task=c1)
    c3 = ReduceTask(
            base_dir=FLAGS.out_dir,
            num_processes=1,
            num_tasks=200,
            previous_task=c2)
    # common_tasks = [c1, c2, c3]
    # common_tasks = [c2, c3]
    common_tasks = [c1]

    # - Duplicate elimination
    d_control = EliminateControlDuplicatesTask(
        base_dir=FLAGS.out_dir,
        num_processes=4,
        num_tasks=200,
        previous_task=common_tasks[-1])
    d_data = EliminateDataDuplicatesTask(
        base_dir=FLAGS.out_dir,
        num_processes=num_cpus,
        num_tasks=200,
        previous_task=common_tasks[-1])

    if FLAGS.eliminate_control_duplicates:
        duplicate_ele_tasks = [d_control]
    elif FLAGS.eliminate_data_duplicates:
        duplicate_ele_tasks = [d_data]
    else:
        pass
        raise Exception('No duplicate elimination mode specified')

    # - Preprocess
    a1 = AnalysisTask(
        base_dir=FLAGS.out_dir,
        num_processes=num_cpus,
        num_tasks=1000,
        previous_task=duplicate_ele_tasks[-1])
    analysis_tasks = [a1]

    # - Stats
    s1 = ComputeStatsTask(
        base_dir=FLAGS.out_dir,
        num_processes=num_cpus,
        num_tasks=100,
        previous_task=duplicate_ele_tasks[-1])
    s2 = ReduceTask(
        base_dir=FLAGS.out_dir,
        num_processes=num_cpus,
        num_tasks=1,
        previous_task=s1)
    s3 = PrintStatsTask(
        base_dir=FLAGS.out_dir,
        num_processes=num_cpus,
        num_tasks=1,
        previous_task=s2)
    stats_tasks = [s1, s2, s3]

    # - Dot
    d1 = DotGraphTask(
        base_dir=FLAGS.out_dir,
        num_processes=1,
        num_tasks=1000,
        previous_task=common_tasks[-1])
    dot_tasks = [d1]

    # Complete pipeline
    if FLAGS.preprocess:
        tasks = common_tasks + duplicate_ele_tasks + analysis_tasks
        tasks = common_tasks + duplicate_ele_tasks
        # tasks = analysis_tasks
    elif FLAGS.stats:
        tasks = stats_tasks
    elif FLAGS.dot:
        tasks = dot_tasks
    else:
        raise Exception('No mode specified')

    # Run tasks
    run_tasks(tasks)


def store(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def get_pickle_files(in_dir):
    return fnmatch.filter(os.listdir(in_dir), "*.pickle")


if __name__ == "__main__":
    app.run(main)
