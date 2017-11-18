import os


IN_WIDTH = 42
IN_HEIGHT = 42
IN_CHANNELS = 6

BATCH_SZ = 640

DROP_RATE = 0.0
MAX_POOL = False

NUM_CONV_LAYERS = 3
CHANNEL_SIZES = [32, 64, 32]
CONV_KERNEL_SIZES = [(8, 8), (4, 4), (3, 3)]
CONV_STRIDES = [(4, 4), (2, 2), (1, 1)]
FC_SIZE = 512

NUM_ACTIONS = # varies per game

LEARN_RATE = 0.01
DAMPING_LAMBDA = 0
MOVING_AVG_DECAY = 0.99
KFAC_MOMENTUM = 0.9
EPSILON = 0.01


#VAL_TIMES = 20 #number of times to vlaidate per epoch

assert len(CHANNEL_SIZES) == len(CONV_KERNEL_SIZES) == len(CONV_STRIDES) == NUM_CONV_LAYERS

GRAPH_DIR = os.path.join(os.getcwd(), "Graph")
MODEL_DIR = os.path.join(os.getcwd(), "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "model")

TRAIN_DATA_PATH = "trainData.npy"
VAL_DATA_PATH = "valData.npy"

TRAIN_LABELS_PATH = "trainLabels.npy"
VAL_LABELS_PATH = "valLabels.npy"



#Dimensions of Network
#(160, 320, 3) - 153,600

#(32, 64, 7) - 14,336

#(8, 16, 20) - 2,560

#(2, 4, 70) - 560

#(1)





#1
# NUM_EPOCHS = 3
# BATCH_SZ = 100
# LEARN_RATE = 0.01

# NUM_CONV_LAYERS = 3
# CHANNEL_SIZES = [7, 20, 70]
# CONV_KERNEL_SIZES = [(15, 15), (8, 8), (4,4)]
# CONV_STRIDES = [(5,5), (4,4), (3, 3)]





#0 try 

# NUM_EPOCHS = 1
# BATCH_SZ = 100
# LEARN_RATE = 0.01

# NUM_CONV_LAYERS = 3
# CHANNEL_SIZES = [7, 20, 70]
# CONV_KERNEL_SIZES = [(15, 15), (10, 10), (4,4)]
# CONV_STRIDES = [(4,4), (3,3), (2, 2)]



# -1 try -- too long to train:

# NUM_EPOCHS = 1
# BATCH_SZ = 100
# LEARN_RATE = 0.01

# NUM_CONV_LAYERS = 5
# CHANNEL_SIZES = [7, 40, 60, 60, 70]
# CONV_KERNEL_SIZES = [(30, 30), (20, 20), (10, 10), (5, 5), (4,4)]
# CONV_STRIDES = [(1,1), (2,2), (2, 2), (2, 2), (2,2)]

#Dimensions of Network
#(160, 320, 3) - 153,600

#(160, 320, 7) - 358,400

#(80, 160, 40) - 512,000

#(40, 80, 60) - 192,000

#(20, 40, 60) - 48,500

#(5, 10, 70) - 3,500

#(1)

