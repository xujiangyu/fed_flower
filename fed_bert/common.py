import os

TOTAL_UPDATES = 31  # Total number of training steps
#TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES = 10  # Warmup the learning rate over this many updates
#WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR = 0.0005  # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE = 512  # Max sequence length
MAX_POSITIONS = 512  # Num. positional embeddings (usually same as above)
MAX_SENTENCES = 16  # Number of sequences per batch (batch size)
UPDATE_FREQ = 16  # Increase the batch size 16x
MAX_EPOCH = 10  # Max local epoch of each client

#ROOT_DIR=./biobert/datasets_clients
ROOT_DIR = "/home/yawei/githubs/FedBERT/datasets_clients"
CLIENTS_NUMBER = 10
AVG_PERIOD = 5
CLIENT_ID = 0

# for epoch in range(0, MAX_EPOCH, AVG_PERIOD):
#     current_iter = epoch + AVG_PERIOD

#     cmd_str = f"fairseq-train --fp16 {ROOT_DIR}/client_{CLIENT_ID}/data-bin --task masked_lm --criterion masked_lm " \
#                     f"--arch roberta_base --sample-break-mode complete --tokens-per-sample {TOKENS_PER_SAMPLE} --optimizer adam " \
#                     f"--adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr {PEAK_LR} --warmup-updates {WARMUP_UPDATES} " \
#                     f"--total-num-update {TOTAL_UPDATES} --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size {MAX_SENTENCES} --update-freq {UPDATE_FREQ} " \
#                     f"--max-update {TOTAL_UPDATES} --log-format simple --log-interval 1 --tensorboard-logdir {ROOT_DIR}/client_{CLIENT_ID}/logdir --save-interval 1 --max-epoch {AVG_PERIOD} " \
#                     f"--save-dir {ROOT_DIR}/client_{CLIENT_ID}/checkpoints --restore-file {ROOT_DIR}/client_{CLIENT_ID}/server/checkpoint_avg.pt"
#     # assert os.system(cmd_str) == 0

#     print(epoch, cmd_str)
