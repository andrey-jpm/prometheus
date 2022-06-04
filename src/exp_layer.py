import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

class ExperimentalLayer():
    """
        This layer calculates observables based on pdf models output

        input_size: dict
            keys are pdf names, elements are lists of input sized for each dataset, which are used to split pdf outputs
        dataset_name: dict
            keys are pdf names, elements are dataset names which we use to relate pdf outputs and corresponding observables class
        obs: dict
            contains observable classes, takes pdf output and calculates prediction for the observable
        target:
            TF tensor with measured datapoints (observables)
    """
    def __init__(self, exp_layer_info):
        self.layer_info = exp_layer_info
        self.input_size = self.layer_info["input_size"]
        self.dataset_name = self.layer_info["dataset_name"]
        self.obs = self.layer_info["obs"]
        self.target = exp_layer_info["target"]
        self.invcov_matrix = exp_layer_info["invcov_matrix"]
        self.training_mask = exp_layer_info["training_mask"]
        self.validation_mask = exp_layer_info["validation_mask"]
        self.kfold_mask = exp_layer_info["kfold_mask"]
        self.ndata = exp_layer_info["ndata"]
        self.name = exp_layer_info["name"]

    def __call__(self, pdf_models):
        dataset_models = {}

        for pdf_name in pdf_models.keys():
            if self.name == "pos_exp":
                dataset_models[pdf_name] = {self.dataset_name[pdf_name][0]: pdf_models[pdf_name]}
            else:
                # split the pdf output into dataset inputs
                dataset_models[pdf_name] = Lambda(tf.split, arguments={"num_or_size_splits": self.input_size[pdf_name], 
                "axis": -2})(pdf_models[pdf_name])
                # transform the output into a dictionary  
                dataset_models[pdf_name] = dict(zip(self.dataset_name[pdf_name], dataset_models[pdf_name]))

        output = {"name": self.name, "chi2_training": None, "chi2_validation": None, "chi2_kfold": None}
        y_pred = []
        # run over all datasets
        for dataset_name in self.obs.keys():
            # obtain observable module for each dataset
            observable = self.obs[dataset_name]
            # obtain pdf outputs for this dataset
            pdf_models = {}
            for pdf_name in dataset_models.keys():
                if dataset_name in dataset_models[pdf_name]:
                    pdf_models[pdf_name] = dataset_models[pdf_name][dataset_name]

            # calculate observables
            output[dataset_name] = observable(pdf_models)
            # update residues
            y_pred += [output[dataset_name]]

        # calculate residues
        y_pred = K.concatenate(y_pred, axis=-1)
        # calculate chi2 for training, validation and kfold sets
        for chi2_name, chi2_mask in [
            ("chi2_training", self.training_mask),
            ("chi2_validation", self.validation_mask),
            ("chi2_kfold", self.kfold_mask),
            ]:
            output[chi2_name] = self.get_variance(self.target, y_pred, chi2_mask)
            output[chi2_name] = tf.expand_dims(output[chi2_name], axis=0)

        # expected shapes of the output["chi2_name"] is (1,1)
        return output

    def get_variance(self, target, y_pred, mask):
        res = (target - y_pred)*mask
        right_dot = tf.tensordot(self.invcov_matrix, res[0,:], axes=1)
        res = tf.tensordot(res, right_dot, axes=1)
        return res