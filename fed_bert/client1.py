from bert import *

CLIENT_ID = 0

client = BertClient(CLIENT_ID=0)

fl.client.start_numpy_client(client=client, server_address="localhost:8092")