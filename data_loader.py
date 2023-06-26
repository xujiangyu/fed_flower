import FedMedMNIST
import os
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from FedMedMNIST import *


def read_data(data_path, dataset_name):
    print('Preparing dataset {}'.format(dataset_name))

    if dataset_name in MedMNIST_DATASETS:

        dataset = getattr(FedMedMNIST, MedMNIST_INFO_DICT[dataset_name]['python_class'])

        # prepocess the dataset
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

        # load the dataset
        train_dataset = dataset(split='train', transform=data_transform, dataset_dir=data_path)
        test_dataset = dataset(split='test', transform=data_transform, dataset_dir=data_path)

        train_num = len(train_dataset)
        test_num = len(test_dataset)

        train_dl = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        test_dl = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        return train_dataset, test_dataset, train_dl, test_dl, train_num, test_num


def load_partition_data_medmnist(data_path, dataset_name, partition, num_clients, batch_size):
    train_imgs = []
    train_targets = []
    test_imgs = []
    test_targets= []
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    train_dataset, test_dataset, train_dl, test_dl, train_num, test_num = read_data(data_path, dataset_name)

    # training data partition
    for train_data in train_dl:
        train_images, train_labels = train_data
        train_targets.append(train_labels[:, 0].numpy())
        for train_image in train_images:
            train_imgs.append(train_image)

    train_targets = torch.LongTensor(np.concatenate(train_targets))
    train_imgs = torch.stack(train_imgs)

    train_global_dataset = Data.TensorDataset(train_imgs, train_targets)
    train_data_global = DataLoader(train_global_dataset, batch_size, shuffle=True)

    # test data partition
    for test_data in test_dl:
        test_images, test_labels = test_data
        test_targets.append(test_labels[:, 0].numpy())
        for test_image in test_images:
            test_imgs.append(test_image)

    test_targets = torch.LongTensor(np.concatenate(test_targets))
    test_imgs = torch.stack(test_imgs)

    test_global_dataset = Data.TensorDataset(test_imgs, test_targets)
    test_data_global = DataLoader(test_global_dataset, batch_size, shuffle=False)

    # split the dataset for federated learning
    fed_class_name = type(train_dataset).__name__ + 'Partitioner'
    fed_train_data = globals()[fed_class_name](targets=train_targets, num_clients=num_clients, partition=partition, dir_alpha=0.5)
    num_classes = fed_train_data.num_classes
    fed_test_data = globals()[fed_class_name](targets=test_targets, num_clients=num_clients, partition=partition, dir_alpha=0.5)
    client_train_dict = fed_train_data.client_dict
    client_test_dict = fed_test_data.client_dict

    fed_file_path = os.path.join(data_path, dataset_name, partition)
    os.makedirs(fed_file_path, exist_ok=True)

    for client_train_id, indices in client_train_dict.items():
        client_dir = os.path.join(fed_file_path, "client_{}".format(client_train_id), "train")
        os.makedirs(client_dir, exist_ok=True)

        client_train_targets = torch.LongTensor([train_targets[i] for i in indices])
        client_train_images = train_imgs[indices]

        client_train_data = Data.TensorDataset(client_train_images, client_train_targets)
        train_data_local_dict[client_train_id] = DataLoader(client_train_data, batch_size, shuffle=True)
        train_data_local_num = len(client_train_data)
        train_data_local_num_dict[client_train_id] = train_data_local_num

    for client_test_id, indices in client_test_dict.items():
        # test data saving
        client_dir = os.path.join(fed_file_path, "client_{}".format(client_test_id), "test")
        os.makedirs(client_dir, exist_ok=True)

        client_test_targets = torch.LongTensor([test_targets[i] for i in indices])
        client_test_images = test_imgs[indices]

        client_test_data = Data.TensorDataset(client_test_images, client_test_targets)
        test_data_local_dict[client_test_id] = DataLoader(client_test_data, batch_size, shuffle=False)

    dataset = (
        train_num,
        test_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_classes
    )
    return dataset, num_classes


if __name__ == "__main__":
    data_path = "~/workspace/downloaded_data/medmnist"
    dataset_name = "breastmnist"
    partition = "noniid-labeldir"
    num_clients = 2
    batch_size = 32
    # train_dataset, test_dataset, train_dl, test_dl, train_num, test_num = read_data(data_path, dataset_name)
    # print(train_num)
    load_partition_data_medmnist(data_path, dataset_name, partition, num_clients, batch_size)