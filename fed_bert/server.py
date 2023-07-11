from bert import *

bertServer = FlowerServer()

fl.server.start_server(server_address="localhost:8092",
                       config=fl.server.ServerConfig(num_rounds=3),
                       strategy=bertServer, grpc_max_message_length=1048576000)