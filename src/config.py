# number of points in the momentum grid
KPTS = 101
if not KPTS % 2: KPTS+=1
KPTS2 = KPTS**2
KPTS_H = KPTS//2 + 1
# maximum momentum value
KMAX = 4
KMAX2 = KMAX**2
# number of flavours
FLAV = 2

# positivity boundaries and number of points in the positivity grid
XPOS_MIN = 0.01
XPOS_MAX = 0.9
POS_DENSITY = 100

# weights for positivity datasets
MAX_LAMBDA = 1e7
LWEIGHTS=[1.0]*FLAV
POS_ALPHA = 1e-7
POS_THRESHOLD = 1e-1

# parameters of the fit
params = {
    "optimizer_name": "Adam",
    "learning_rate": 0.01,
    "initializer": "glorot_normal",
    "max_epochs": 5000,
    "stopping_patience": 0.1,
    "replicas": 10,
    "nodes": [20, FLAV],
    "activations": ['tanh', 'linear'],
    "nn_depth" : 3,
    "nn_type": "DENSE_FACT",
    "positivity": True,
    "pos_type": "FPOS",
    "n_kfolds": 1,
    "validation_size": 0.25,
    "lambda_undate_freq": 100,
    }

# list of experimental dictionaries with dataset files
exp_dict_list = [
    {"exp_name": "HERMES_2002",
    "type": "SIDIS",
    "dataset_names": ["2002.csv"],
    "invcov_matrix": None},
]
