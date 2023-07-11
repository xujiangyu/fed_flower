from bert import *
import torch

model = torch.load("model.pt", map_location=torch.device("cpu"))

client = BertClient(CLIENT_ID=1, model=model)

fl.client.start_numpy_client(client=client, server_address="localhost:8092")