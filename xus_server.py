import os
import time
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import flwr as fl
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from flwr.common import Metrics, EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from utils.logs import Logger
from config import task, model, DEVICE, server_address, num_rounds, model_path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net

if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net

if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net

net = Net().to(DEVICE)

# 实例化训练日志
logpath = './logs/log/'
if not os.path.exists(logpath):
    os.makedirs(logpath)
log_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
logname = "federated_log" + "_" + task + "_" + model + "_" + str(log_time) + "_log.txt"
logfile = os.path.join(logpath, logname)
log = Logger(logfile, level='info')

# 可视化
visual_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
visual_path = os.path.join("logs", "visualization", str(visual_time))
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
writer = SummaryWriter(visual_path)

# fl.server.strategy.FedAvg

class SaveModelStrategy(fl.server.strategy.FedAvg):
    # 聚合训练
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        num_client = 0
        for _, res in results:
            print("res.metrics: ", res.metrics)
            accuracy = float(list(res.metrics.values())[0])
            loss = float(str(list(res.metrics.keys())[0]).split("_")[0].split("=")[-1])
            distributed_loss = "training_distributed_loss" + "_client_" + str(num_client)
            distributed_accuracy = "training_distributed_accuracy" + "_client_" + str(num_client)
            print("round: {}/{},   {}: {},  {}: {}".format(
                server_round,
                num_rounds,
                distributed_loss,
                loss,
                distributed_accuracy,
                accuracy))

            writer.add_scalar(distributed_loss, loss, server_round)
            writer.add_scalar(distributed_accuracy, accuracy, server_round)

            num_client += 1

        training_aggregated_loss = weighted_train_loss_avg([(res.num_examples,
                                                             float(str(list(res.metrics.keys())[0]).split("_")[0].split("=")[-1]))
                                                            for _, res in results])

        training_aggregated_accuracy = float(list(aggregated_metrics.values())[0])
        print("round: {}/{},   training_aggregated_loss: {},   training_aggregated_accuracy: {}".format(server_round,
                                                                                                        num_rounds,
                                                                                                        training_aggregated_loss,
                                                                                                        training_aggregated_accuracy))

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters..." + "\n")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            save_path = "./save_model"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #torch.save(net.state_dict(), os.path.join(save_path, f"model_round_{server_round}_{task}_{model}.pth"))
            torch.save(net.state_dict(), model_path)

        log.logger.info(
            "round: {}/{},   training_aggregated_loss: {},   training_aggregated_accuracy: {}".format(server_round,
                                                                                                      num_rounds,
                                                                                                      training_aggregated_loss,
                                                                                                      training_aggregated_accuracy))
        writer.add_scalar('training_aggregated_loss', training_aggregated_loss, server_round)
        writer.add_scalar('training_aggregated_accuracy', training_aggregated_accuracy, server_round)

        return aggregated_parameters, aggregated_metrics

    # 聚合测试
    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        num_client = 0
        for _, res in results:
            distributed_loss = "test_distributed_loss" + "_client_" + str(num_client)
            distributed_accuracy = "test_distributed_accuracy" + "_client_" + str(num_client)
            print("round: {}/{},   {}: {},   {}: {}".format(server_round, num_rounds, distributed_loss, res.loss,
                                                            distributed_accuracy, res.metrics['accuracy']))
            writer.add_scalar(distributed_loss, res.loss, server_round)
            writer.add_scalar(distributed_accuracy, res.metrics['accuracy'], server_round)

            num_client += 1

        print("round: {}/{},   test_aggregated_loss: {},   test_aggregated_accuracy: {}".format(server_round,
                                                                                                num_rounds,
                                                                                                aggregated_loss,
                                                                                                aggregated_metrics[
                                                                                                    'accuracy']))

        log.logger.info("round: {}/{},   test_aggregated_loss: {},   test_aggregated_accuracy: {}".format(server_round,
                                                                                                          num_rounds,
                                                                                                          aggregated_loss,
                                                                                                          aggregated_metrics[
                                                                                                              'accuracy']))
        writer.add_scalar('test_aggregated_loss', aggregated_loss, server_round)
        writer.add_scalar('test_aggregated_accuracy',  aggregated_metrics['accuracy'], server_round)

        return aggregated_loss, aggregated_metrics


def weighted_train_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(list(m.values())[0]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def evaluate_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    start_time = time.time()
    # Create strategy and run server
    strategy = SaveModelStrategy(fit_metrics_aggregation_fn=fit_weighted_average,
                                 evaluate_metrics_aggregation_fn=evaluate_weighted_average)
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds),
        strategy=strategy
    )
    end_time = time.time()
    federated_time = end_time - start_time
    log.logger.info("Federated Training Time: {}".format(federated_time))


if __name__ == "__main__":
    main()