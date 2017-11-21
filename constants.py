IN_WIDTH = 42
IN_HEIGHT = 42
IN_CHANNELS = 6

NUM_CONV_LAYERS = 3
CHANNEL_SIZES = [32, 64, 32]
CONV_KERNEL_SIZES = [(8, 8), (4, 4), (3, 3)]
CONV_STRIDES = [(4, 4), (2, 2), (1, 1)]
FC_SIZE = 512

assert len(CHANNEL_SIZES) == len(CONV_KERNEL_SIZES) == len(CONV_STRIDES) == NUM_CONV_LAYERS

ENTROPY_REGULARIZATION_WEIGHT = 0.01

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

