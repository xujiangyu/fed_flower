from bert import *

bertServer = FlowerServer()

fl.server.start_server(server_address="localhost:8092",
                       config=fl.server.ServerConfig(num_rounds=3),
                       strategy=bertServer)