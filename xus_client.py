import os
import numpy as np
from collections import OrderedDict
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from config import *
from tqdm import tqdm
import utils.loss
from utils.logs import Logger
from sklearn.metrics import accuracy_score, roc_auc_score
import flwr as fl


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def getAUC(y_true, y_score, task):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


# 实例化训练日志
log_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logpath = './logs/log/'
if not os.path.exists(logpath):
    os.makedirs(logpath)
logname = os.path.basename(__file__).split(
    ".")[0] + "_" + task + "_" + model + "_" + str(log_time) + "_log.txt"
logfile = os.path.join(logpath, logname)
log = Logger(logfile, level='info')

warnings.filterwarnings("ignore", category=UserWarning)

if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net

if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net

if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net

if data_random_split:
    from utils.dataset import load_random_split_data as load_data
    trainloader, testloader, num_examples = load_data(data_path)

else:
    from utils.dataset import load_data_from_path as load_data
    trainloader, testloader, num_examples = load_data(trainset, testset)


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    # 损失函数
    global criterion, optimizer
    if loss_func == "cross_entropy_3d":
        criterion = utils.loss.CrossEntropy3D().to(DEVICE)

    if loss_func == "dice_loss":
        n_classes = num_classes
        criterion = utils.loss.DiceLoss(n_classes).to(DEVICE)

    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if loss_func == "mse":
        criterion = torch.nn.MSELoss().to(DEVICE)

    # 优化器
    if optimiz == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps,
                                     weight_decay, amsgrad)

    if optimiz == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum, dampening,
                                    weight_decay, nesterov)

    # 评价参数初始化
    correct, total, loss, accuracy, running_loss_ = 0, 0, 0.0, 0.0, 0.0

    y_true = np.array([])
    y_score = np.array([])
    y_score = None

    # 迭代训练
    for _ in range(0, epochs):
        # 当前训练轮次loss初始化：
        running_loss = 0.0

        for images, labels in tqdm(trainloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            #            print("images shape: ", images.shape)
            # images = images.half()

            outputs = net(images)
            print("================", outputs.shape)
            pred = torch.argmax(outputs.data, dim=1)
            print("=======pred========", pred.shape)
            loss = criterion(outputs, labels)
            running_loss += loss
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            y_true = np.append(y_true, labels.cpu().detach().numpy())
            if y_score is None:
                y_score = outputs.cpu().detach().numpy().reshape(outputs.shape)
            else:
                y_score = np.vstack([
                    y_score,
                    outputs.cpu().detach().numpy().reshape(outputs.shape)
                ])
            print(y_true.shape, y_score.shape)

            del outputs, pred, images, labels
            torch.cuda.empty_cache()
            for para in net.parameters():
                print("net model params: ", para)

        # y_true = y_true.numpy()
        # y_score = y_score.detach().numpy()

        running_loss_ = running_loss / len(trainloader)

        if task == "segmentation":
            accuracy = correct / (total * img_size[0] * img_size[1] *
                                  img_size[2])
            auc = 0.0

        if task == "classification":
            accuracy = correct / total
            try:
                auc = getAUC(y_true, y_score, task=class_number_type)
            except:
                auc = 0.0

        print(
            "epoch {}/{},   training loss: {},   training accuracy: {}".format(
                str(_ + 1), str(epochs), running_loss_, accuracy))
        log.logger.info(
            "epoch: {}/{},   training loss: {},   training accuracy: {}".format(
                str(_ + 1), str(epochs), running_loss_, accuracy))

    return running_loss_, accuracy, auc


def test(net, testloader):
    """Validate the network on the entire test set."""
    # 损失函数
    global criterion
    if loss_func == "cross_entropy_3d":
        criterion = utils.loss.CrossEntropy3D().to(DEVICE)

    if loss_func == "dice_loss":
        n_classes = num_classes
        criterion = utils.loss.DiceLoss(n_classes).to(DEVICE)

    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if loss_func == "mse":
        criterion = torch.nn.MSELoss().to(DEVICE)
    # 评价参数初始化
    correct, total, loss, accuracy = 0, 0, 0.0, 0.0
    y_true = np.array([])
    y_score = None
    # 迭代测试
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # images = images.half()
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs.data, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            y_true = np.append(y_true, labels.cpu().detach().numpy())
            if y_score is None:
                y_score = outputs.cpu().detach().numpy().reshape(outputs.shape)
            else:
                y_score = np.vstack([
                    y_score,
                    outputs.cpu().detach().numpy().reshape(outputs.shape)
                ])
            print(y_true.shape, y_score.shape)

            del data, images, labels, pred, outputs

    loss = loss / len(testloader)

    if task == "segmentation":
        accuracy = correct / (total * img_size[0] * img_size[1] * img_size[2])
        auc = 0.0

    if task == "classification":
        accuracy = correct / total
        #acc = getACC(y_true, y_score, task=class_number_type)

        try:
            auc = getAUC(y_true, y_score, task=class_number_type)
        except:
            auc = 0.0

    return loss, accuracy, auc


# Load model and data (DenseNet, Local Data)

#net = Net().half().to(DEVICE)
# net = Net().to(DEVICE)
#net.eval()

net = Net()
# net.half()
# net.cuda()
# net.eval()


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, auc = train(net, trainloader, epochs)

        return self.get_parameters(config={}), len(trainloader.dataset), {
            "loss=" + str(float(auc)) + "_" + "accuracy": accuracy
        }

        # return self.get_parameters(config={}), len(trainloader.dataset), {tuple([loss, auc]): accuracy}
        # return self.get_parameters(config={}), len(trainloader.dataset), {"loss": loss,  "accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # param_path = os.path.join(save_path, f"model_round_{server_round}_{task}_{model}.pth")
        # param_path = f"./save_model/model_round_99_classification_DenseNet.pth"
        param_path = model_path
        #net.load_state_dict(torch.load(param_path))

        loss, accuracy, auc = test(net, testloader)
        return auc, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address=server_address,
    client=FlowerClient(),
)