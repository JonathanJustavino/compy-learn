import dgl.nn.pytorch
import torch
import math
import torch.nn.functional as F
import numpy as np

from torch import nn
from dgl import function as dglF
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from compy.models.model import Model


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        annotation_size = config["hidden_size_orig"]                             # annotation size: 8
        hidden_size = config["gnn_h_size"]
        n_steps = config["num_timesteps"]
        n_etypes = config["num_edge_types"]
        branch_count = 1

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.gconv = GatedGraphConv(hidden_size, n_steps)
        self.lin = nn.Linear(hidden_size, branch_count)

    def forward(
            self, graph, dimension=0,
    ):
        x, edge_index, batch, index = graph.x, graph.edge_index, graph.batch, graph.source_nodes
        indices = [node_idx for sublist in index for node_idx in sublist]
        idx = self.__filter_branch_pairs(indices)
        idx = torch.LongTensor(idx)

        x = self.reduce(x)
        x = self.gconv(x, edge_index)

        x = torch.index_select(x, dimension, idx)
        x = self.lin(x)
        x = torch.sigmoid(x)

        return x

    def __filter_branch_pairs(self, probabilities):
        filtered = []
        if len(probabilities) == 0:
            return filtered
        previous = probabilities[-1]
        for index in probabilities:
            if index == previous:
                continue
            filtered.append(index)
            previous = index
        return filtered


class GnnPytorchBranchProbabilityModel(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "num_timesteps": 4,
                "hidden_size_orig": num_types,
                "gnn_h_size": 64,
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 1000,
                "num_edge_types": 1,
            }
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(config)
        self.model = self.model.to(self.device)

    def __process_data(self, data):
        return [
            {
                "nodes": data["x"]["code_rep"].get_node_list(),
                "edges": data["x"]["code_rep"].get_edge_list_tuple(),
                "probability": data["x"]["code_rep"].get_edge_list_with_data(),
            }
            for data in data
        ]

    def __build_pg_graphs(self, batch_graphs):
        pg_graphs = []
        total_node_count = 0

        # Graph
        for graph_index, batch_graph in enumerate(batch_graphs):
            # Nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1
            x = torch.tensor(one_hot, dtype=torch.float)

            # Edges
            edge_index, edge_features, probability_list, source_nodes = [], [], [], []
            probability = "probability"
            for index, edge in enumerate(batch_graph["edges"]):
                last_element = batch_graph[probability][index][-1]
                edge_type = batch_graph[probability][index][1]
                source_node = edge[0]
                # edge_type = edge[1]
                edge_index.append([source_node, edge[2]])
                edge_features.append(edge_type)             # edge type

                if probability in last_element and edge_type == 5:
                    source_nodes.append(source_node)
                    probability_list.append(last_element[probability])

            # Probability Nodes
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.long)

            node_count = len(batch_graphs[graph_index - 1]["nodes"]) if graph_index > 0 else 0
            total_node_count += node_count
            source_nodes = [source_node + total_node_count for source_node in source_nodes]

            #FIXME was ist das Problem hier, warum macht es hier nicht sinn, das zweimalige aufkommen eines indizes
            # hintereinander, herauszufiltern
            # probability_list = self._filter_branch_pairs(probability_list)
            edge_probabilities = self.__get_probability_tensor(probability_list)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                source_nodes=source_nodes,
                y=edge_probabilities
            )

            pg_graphs.append(graph)

        return pg_graphs

    def __filter_branch_pairs(self, probabilities):
        filtered = []
        if len(probabilities) == 0:
            return filtered
        previous = probabilities[-1]
        for index in probabilities:
            if index == previous:
                continue
            filtered.append(index)
            previous = index
        return filtered

    def __get_probability_tensor(self, probabilities, skip_pair=False):
        edge_probabilities = []
        length = len(probabilities)
        for index, prob in enumerate(probabilities):
            if skip_pair:
                skip_pair = False
                continue
            if prob == 100:
                edge_probabilities.append([prob, 0])
            if prob == 0:
                edge_probabilities.append([prob, 100])
            elif index + 1 < length:
                pair_prob = probabilities[index + 1]
                total_probability = prob + pair_prob
                if 99 <= total_probability <= 100:
                    edge_probabilities.append([prob, pair_prob])
                    skip_pair = True
        return torch.tensor(edge_probabilities, dtype=torch.float) / 100

    def _train_init(self, data_train, data_valid):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        return self.__process_data(data_train), self.__process_data(data_valid)

    def _train_with_batch(self, batch):
        batch_size = 999999
        graph = self.__build_pg_graphs(batch)
        loader = DataLoader(graph, batch_size=batch_size)
        batch_loss = 0
        correct_sum = 0

        for data in loader:
            data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            pred = self.model(data)
            pred_left = pred[:, 0]
            truth = data.y[:, 0]
            loss = F.mse_loss(pred_left, truth)

            loss.backward()
            self.opt.step()

            batch_loss += loss
            correct_sum += pred.max(dim=1)[1].eq(truth.view(-1)).sum().item()

        train_accuracy = correct_sum / len(loader.dataset)
        train_loss = batch_loss / len(loader.dataset)
        return train_loss, train_accuracy

    def _predict_with_batch(self, batch):
        correct = 0
        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)
        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)
                truth = data.y[:, 0] if data.y.nelement() > 0 else data.y

            correct += pred.max(dim=1)[1].eq(truth.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)

        return valid_accuracy, pred
