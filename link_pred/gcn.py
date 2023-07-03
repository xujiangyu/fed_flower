import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Callable, Dict, List, Optional, Tuple
import flwr as fl
from collections import OrderedDict
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar


class GCNLinkPred(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GCNLinkPred, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def get_link_labels(pos_edge_index, neg_edge_index, device):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.0
    return link_labels


def train(model, train_loader, args):
    model.to(args["device"])
    model.train()

    if args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=args["learning_rate"],
                                    weight_decay=args["weight_decay"])
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )

    for epoch in range(args["epochs"]):
        for idx_batch, batch in enumerate(train_loader):
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1),
            )

            z = model.encode(batch.x, batch.edge_index)
            train_z = z

            edge_idx, neg_idx = batch.edge_index.to(
                args["device"]), neg_edge_index.to(args["device"])

            link_logits = model.decode(z, edge_idx, neg_idx)
            link_labels = get_link_labels(edge_idx, neg_idx, args["device"])
            loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
            loss.backward()
            optimizer.step()


def test(model, test_loader, args):
    model.eval()
    model.to(args["device"])

    cum_score = 0.0
    ngraphs = 0
    threshold = torch.tensor([0.7], device=args["device"])
    for batch in test_loader:
        batch.to(args["device"])
        with torch.no_grad():

            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_index.size(1),
            )
            z = model.encode(batch.x, batch.edge_index)
            out = model.decode(z, batch.edge_index,
                               neg_edge_index).view(-1).sigmoid()
            pred = (out > threshold).float() * 1

        cum_score += average_precision_score(np.ones(batch.edge_index.numel()),
                                             pred.cpu())
        ngraphs += batch.num_graphs

    return cum_score / ngraphs, model


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self,
                 cid,
                 net,
                 trainloader,
                 valloader=None,
                 args=None) -> None:
        super().__init__()
        self.cid = cid
        self.net = net
        self.trainloader = trainloader[self.cid]
        self.valloader = valloader
        self.args = args

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.net)

    def fit(self, parameters: NDArrays,
            config: Dict[str,
                         Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters=parameters)
        train(self.net, self.trainloader, self.valloader, self.args['device'],
              self.args)

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(
            self, parameters: NDArrays,
            config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters)

        if self.valloader is None:
            return 0.0, 20, {"accuracy": 0.95}

        score, model = test(self.net, self.valloader, self.args['device'],
                            self.args)
        return score, len(self.valloader), {"accuracy": 0.95}


class FlowerServer(fl.server.strategy.FedAvg):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
                                       Optional[Tuple[float,
                                                      Dict[str,
                                                           Scalar]]],]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str,
                                                             Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)


args = {}
args["device"] = "cpu"
args["optimizer"] = "sgd"
args["learning_rate"] = 0.05
args["epochs"] = 20
args["weight_decay"] = 0.01
args["metric"] = "auc"
args["nfeat"] = 602
args["nclass"] = 6
args["nlayer"] = 5
args["dropout"] = 0.3
args["nhid"] = 0.3
args["in_channels"] = 602
args["out_channels"] = 6
args["trainloader"] = None