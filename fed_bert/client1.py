from bert import *
import torch

model = torch.load("model.pt", map_location=torch.device("cpu"))
CLIENT_ID = 0

client = BertClient(CLIENT_ID=0, model=model)

fl.client.start_numpy_client(client=client,
                             server_address="localhost:8092",
                             grpc_max_message_length=1048576000)
