from fed_tabnet import *
from data_loader import test_tabnet

data_path = "complex_disease.csv"
x0, x1, y0, y1, test_x, test_y = test_tabnet(data_path)

model = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                         cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
                         n_d=8,
                         n_a=8)

client = TabNetClient(model, test_x, test_y, x1, y1)

fl.client.start_numpy_client(client=client, server_address="localhost:8092")
