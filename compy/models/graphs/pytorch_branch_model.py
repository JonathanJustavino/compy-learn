import dgl.nn.pytorch
import torch
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
        n_branches = 2

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.gconv = GatedGraphConv(hidden_size, n_steps)
        self.lin = nn.Linear(hidden_size, n_branches)

    def forward(
            self, graph,
    ):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        print(x.shape)                                                           # [input size: ?, annotationsize: 8]

        x = self.reduce(x)

        print('after reduce', x.shape)                                           # [input size: 978, annotationsize: 32]

        x = self.gconv(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)

        x = F.log_softmax(x, dim=1)

        print('after softmax', x.shape)                                          # [input size: 978, annotationsize: 2]

        return x


class TrialNet(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super(Net, self).__init__()
        self.conv1 = dgl.nn.pytorch.SAGEConv(in_feats=in_features, out_feats=hidden, aggregator_type='mean')
        self.conv2 = dgl.nn.pytorch.SAGEConv(in_feats=hidden, out_feats=out_features, aggregator_type='mean')

    def forward(self, graph, inputs):
        with graph.local_scope():
            graph.ndata['h'] = inputs
            graph.apply_edges(dglF.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']
            inputs = self.conv1(graph, inputs)
            inputs = F.relu(inputs)
            inputs = self.conv2(inputs)
            return inputs


class GnnPytorchBranchProbabilityModel(Model):
    # fixme dimensions anpassen, vielleicht one hot encoding fehler? 978 != 108
    # fixme entweder one hot encoding überarbeiten (unwahrscheinlich), oder netzarchitektur nochmal überarbeiten

    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "num_timesteps": 4,
                "hidden_size_orig": num_types,
                "gnn_h_size": 32,
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
        print("len batch graphs", len(batch_graphs))
        counter = 0

        # Graph
        for batch_graph in batch_graphs:
            # Nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1
            x = torch.tensor(one_hot, dtype=torch.float)

            # Edges
            edge_index, edge_features, edge_probabilities = [], [], []
            prob = "probability"
            for index, edge in enumerate(batch_graph["edges"]):
                last_element = batch_graph[prob][index][-1]
                edge_index.append([edge[0], edge[2]])
                edge_features.append([edge[1]])             # edge type
                if prob in last_element:
                    edge_probabilities.append(last_element[prob])

            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.long)

            #TODO: anpassen
            edge_probabilities = torch.tensor(edge_probabilities, dtype=torch.float)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                y=edge_probabilities
            )

            pg_graphs.append(graph)

        print(len(pg_graphs))

        return pg_graphs

    def _train_init(self, data_train, data_valid):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        return self.__process_data(data_train), self.__process_data(data_valid)

    def _train_with_batch(self, batch):
        batch_size = 100
        graph = self.__build_pg_graphs(batch)
        loader = DataLoader(graph, batch_size=batch_size)
        batch_loss = 0
        correct_sum = 0

        for data in loader:
            data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            print("\ndata", data)

            pred = self.model(data)

            print("\n shape", pred.shape)
            print("\n shape", data.y.shape)


            loss = F.nll_loss(pred, data.y)
            loss.backward()
            self.opt.step()

            batch_loss += loss
            correct_sum += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

        train_accuracy = correct_sum / len(loader.dataset)
        train_loss = batch_loss / len(loader.dataset)

        return train_loss, train_accuracy

    def _predict_with_batch(self, batch):
        correct = 0
        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=100)
        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)

            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)

        return valid_accuracy, pred
