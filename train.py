import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import warnings
from config import *
from tqdm import tqdm
import utils.loss
from utils.logs import Logger

# 实例化训练日志
log_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
logpath = './logs/log/'
if not os.path.exists(logpath):
    os.makedirs(logpath)
logname = os.path.basename(__file__).split(".")[0] + "_" + task + "_" + model + "_" + str(log_time) + "_log.txt"
logfile = os.path.join(logpath, logname)
log = Logger(logfile, level='info')

# 可视化
visual_time = time.strftime("%Y%m%d%H%M%S",  time.localtime())
visual_path = os.path.join("logs", "visualization", str(visual_time))
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
writer = SummaryWriter(visual_path)


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
        optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps, weight_decay, amsgrad)

    if optimiz == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum, dampening, weight_decay, nesterov)

    # 评价参数初始化
    correct, total, loss, accuracy = 0, 0, 0.0, 0.0
    # 迭代训练
    for _ in range(0, epochs):
        # 当前训练轮次loss初始化：
        running_loss = 0.0

        for images, labels in tqdm(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            pred = torch.argmax(outputs.data, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss_ = running_loss / len(trainloader)

        if task == "segmentation":
            accuracy = correct / (total * img_size[0] * img_size[1] * img_size[2])

        if task == "classification":
            accuracy = correct / total

        print("epoch {}/{},   training loss: {},   training accuracy: {}".format(str(_ + 1), str(epochs), running_loss_,
                                                                                 accuracy))
        log.logger.info(
            "round: {}/{},   training loss: {},   training accuracy: {}".format(str(_ + 1), str(epochs), running_loss_,
                                                                                accuracy))

        fig_name_loss = os.path.basename(__file__).split(".")[0] + "_" + "train_loss"
        fig_name_accuracy = os.path.basename(__file__).split(".")[0] + "_" + "train_accuracy"
        writer.add_scalar(fig_name_loss, running_loss_, _ + 1)
        writer.add_scalar(fig_name_accuracy, accuracy, _ + 1)


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
    # 迭代测试
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs.data, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    loss = loss / len(testloader)

    if task == "segmentation":
        accuracy = correct / (total * img_size[0] * img_size[1] * img_size[2])

    if task == "classification":
        accuracy = correct / total

    return loss, accuracy


def main():
    net = Net().to(DEVICE)
    train(net, trainloader, epochs=epochs)
    save_path = "./save_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(net.state_dict(), os.path.join(save_path, f"train_model_{task}_{model}.pth"))
    loss, accuracy = test(net, testloader)
    print("测试损失： ", loss)
    print("测试精度为： ", accuracy)


if __name__ == "__main__":
    main()
