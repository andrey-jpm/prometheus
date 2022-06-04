from gc import callbacks
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from datasets import get_exp_info
import pdf
from callbacks import ValidationCheckCallback, UpdateMultipliersCallback

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers as Kopt
from config import POS_DENSITY, XPOS_MIN, XPOS_MAX, FLAV, KPTS_H
from grid import POS_INPUT_GRID, KGRID2
from phconst import flav as flav_list

optimizers = {
    "RMSprop": [Kopt.RMSprop, {"clipnorm": 1.0}],
    "Adam": [Kopt.Adam, {"clipnorm": 1.0}],
    "Adagrad": [Kopt.Adagrad, {"clipnorm": 1.0}],
    "Adadelta": [Kopt.Adadelta, {"clipnorm": 1.0}],
    "Adamax": [Kopt.Adamax, {"clipnorm": 1.0}],
    "Nadam": [Kopt.Nadam, {"clipnorm": 1.0}],
    "Amsgrad": [Kopt.Adam, {"clipnorm": 1.0, "amsgrad": True}],
    "SGD": [Kopt.SGD, {"clipnorm": 1.0, "momentum": 0.0, "nesterov": False}],
}


class Fitter():
    def __init__(self, exp_data, params):
        # dictionary experimental data
        self.exp_data = exp_data
        # list of pdfs we need
        self.pdf_type = []
        # keys are names of pdfs, elements are list of dataset inputs, expected shape (1, ndata*KPTS2, 2)
        self.pdf_input = {}
        # number of points in each experiment
        self.total_input_size = {}
        # names of experiments in the output of each pdf
        self.pdfout_exp_name = {}
        # experiental layers which take pdf outputs and generate data predictions
        self.exp_layers = {}
        self.total_ndata = None

        # target values of observables to fit
        self.target = {}

        # predictions of the model for observables
        self.obs_pred = None
        # predictions for pdfs
        self.pdf_pred = {}

        # statistics of the fit
        self.chi2pp_validation = None
        self.chi2pp_kfold = None

        # pdf models and corresponding tensor inputs
        self.pdf_model = {}
        self.input_ktensor = {}
        self.input_tftensor = {}

        # parameters of the fit
        self.params = params
        self.max_epochs = self.params["max_epochs"]
        self.stopping_patience = self.params["stopping_patience"]
        self.lambda_update_freq = self.params["lambda_undate_freq"]
        self.optimizer_name = self.params["optimizer_name"]
        self.learning_rate = self.params["learning_rate"]

        # fill the dictionaries with experimental data
        self._fill_dictionaries()

    def _fill_dictionaries(self):
        # create a positivity dictionary if needed
        if self.params["positivity"]:
            invcov_matrix = np.ones(shape=(POS_DENSITY,))
            pos_dict = {
                "name": "pos_exp",
                "datasets": [{"name": "pos_dataset", "ndata": POS_DENSITY, 
                            "pdf_type": self.pdf_type, "obs": "positivity", "pos_type": self.params["pos_type"]}],
                "invcov_matrix": np.diag(invcov_matrix),
                "ndata": POS_DENSITY,
                "training_mask": np.ones(shape=(POS_DENSITY,)),
                "validation_mask": np.zeros(shape=(POS_DENSITY,)),
                "kfold_mask": np.zeros(shape=(POS_DENSITY,)),
                }
            self.pos_data = [pos_dict]
        else:
            self.pos_data = []

        self.total_ndata = 0

        # extract relevant information from experimental dictionaries and generate positivity datasets
        for exp_dict in self.exp_data + self.pos_data:
            # parse dictionaries with experimental information into exp_info dictionaries
            exp_info = get_exp_info(exp_dict)

            for pdf_name in exp_info["pdf_input"].keys():
                # add new types of pdfs to the list of pdf inputs self.pdf_input
                if pdf_name not in self.pdf_input.keys():
                    self.pdf_input[pdf_name] = exp_info["pdf_input"][pdf_name]
                    self.total_input_size[pdf_name] = exp_info["total_input_size"][pdf_name]
                    self.pdfout_exp_name[pdf_name] = exp_info["exp_name"][pdf_name]
                    self.pdf_type += [pdf_name]
                else:
                    self.pdf_input[pdf_name] += exp_info["pdf_input"][pdf_name]
                    self.total_input_size[pdf_name] += exp_info["total_input_size"][pdf_name]
                    self.pdfout_exp_name[pdf_name] += exp_info["exp_name"][pdf_name]

            # update target dataset
            self.target.update(exp_info["dataset_target"])
            self.exp_layers.update(exp_info["exp_layer"])

            if exp_dict["name"] != "pos_exp": self.total_ndata += exp_dict["ndata"]

        self.target["chi2_training"] = K.constant(np.zeros(shape=(1,)))
        self.target["chi2_validation"] = K.constant(np.zeros(shape=(1,)))
        self.target["chi2_kfold"] = K.constant(np.zeros(shape=(1,)))

        # concatenate all dataset inputs into a tensor of a shape (1, None, 2)
        for pdf_name in exp_info["pdf_input"].keys():
            self.pdf_input[pdf_name] = np.concatenate(self.pdf_input[pdf_name], axis=-2)

    def _construct_pdf_models(self):
        models = {}

        for pdf_name in self.pdf_type:
            # convert pdf_model into a Layer. input_ktensor contains experimental points
            models[pdf_name] = self.pdf_model[pdf_name](self.input_ktensor[pdf_name])
            # split output between different experiments
            models[pdf_name] = Lambda(tf.split, arguments={"num_or_size_splits": self.total_input_size[pdf_name], 
            "axis": -2})(models[pdf_name])
            # create a dictionary with an experiment number and corresponding output of the pdf_model
            models[pdf_name] = dict(zip(self.pdfout_exp_name[pdf_name], models[pdf_name]))
            # for the positivity observable we need the full information about collinear and transverse parts of the tmd
            if self.params["positivity"]:
                models[pdf_name]["pos_exp"] = self.pdf_model[pdf_name].predict(self.input_ktensor[pdf_name])

        # each pdf model contains a dictionary with the name of the experiment and corresponding output of the pdf_model
        return models

    def _construct_observables(self, models):
        output = {"chi2_training": [], "chi2_validation": [], "chi2_kfold": []}
        chi2_list = ["chi2_training", "chi2_validation", "chi2_kfold"]
        for exp_name in self.exp_layers.keys():
            # read experimental layer
            exp_layer = self.exp_layers[exp_name]
            pdf_models = {}
            # obtain all relevant pdfs for the experimental layer
            for pdf_name in models.keys():
                if exp_name in models[pdf_name]:
                    pdf_models[pdf_name] = models[pdf_name][exp_name]
            # run experimental layer with corresponding pdf output
            exp_layer_output = exp_layer(pdf_models)
            for chi2_name in chi2_list:
                output[chi2_name] += [exp_layer_output[chi2_name]]
                exp_layer_output.pop(chi2_name)
            exp_layer_output.pop("name")
            output.update(exp_layer_output)

        # concatenate results of different experiments
        for chi2_name in chi2_list: 
            output[chi2_name] = K.concatenate(output[chi2_name], axis=-1)
            # calculate total residue summing over all available points
            output[chi2_name] = K.sum(output[chi2_name],axis=-1)

        # return output of the replica
        return output

    def run(self):
        K.clear_session()
        
        for pdf_name in self.pdf_type:
            # generate pdf models and corresponding inputs
            nodes = self.params["nn_depth"]*[self.params["nodes"][0]] + self.params["nodes"][1:]
            activations = self.params["nn_depth"]*[self.params["activations"][0]] + self.params["activations"][1:]
            initializer = self.params["initializer"]

            pdf_model, self.input_ktensor[pdf_name], self.input_tftensor[pdf_name] = pdf.get_pdf_nn(
                nodes,
                activations,
                initializer,
                pdf_name,
            )

            # fill out input tensors with experimental data
            for key in self.input_tftensor[pdf_name].keys():
                if self.input_tftensor[pdf_name][key] == None:
                    if "pdf_inp" in key:
                        self.input_tftensor[pdf_name][key] = K.constant(self.pdf_input[pdf_name])
                    elif "aux_inp" in key:
                        self.input_tftensor[pdf_name][key] = K.constant(POS_INPUT_GRID)
                    self.input_ktensor[pdf_name][key] = Input(shape=self.input_tftensor[pdf_name][key].shape[1:], batch_size=1)

            # FullPDFModel class is a wrapper which contains the collinear part
            self.pdf_model[pdf_name] = pdf.FullPDFModel(pdf_model, pdf_name, self.input_tftensor[pdf_name])

        # convert pdf_model into a Layer by applying it to the experimental points.
        # Each model is a dictionary with a name of the experiment and corresponding output of the pdf model
        pdf_models = self._construct_pdf_models()

        # use outputs of the pdfs to calculate observables
        output = self._construct_observables(pdf_models)

        model = Model(self.input_ktensor, output)

        def loss(y_true, y_pred):
            return y_pred

        opt_args = optimizers[self.optimizer_name][1]
        opt_args["learning_rate"] = self.learning_rate
        optimizer = optimizers[self.optimizer_name][0](**opt_args)

        model.compile(optimizer=optimizer, loss={"chi2_training": loss})

        callbacks_list = [
            ValidationCheckCallback(model, self.input_tftensor, self.max_epochs, self.stopping_patience), 
            UpdateMultipliersCallback(self.max_epochs, self.lambda_update_freq),
            ]

        history = model.fit(x=self.input_tftensor, y=self.target, epochs=self.max_epochs, callbacks=callbacks_list)
        loss_dict = history.history

        # calculate predictions for observables
        self.obs_pred = model.predict(x=self.input_tftensor)
        self.chi2pp_validation = (self.obs_pred["chi2_validation"] / self.total_ndata)[0]
        self.chi2pp_kfold = (self.obs_pred["chi2_kfold"] / self.total_ndata)[0]

        # calculate predictions for pdfs
        for pdf_name in self.pdf_model.keys():
            self.pdf_pred[pdf_name] = {}
            pdf_prediction = self.pdf_model[pdf_name].predict(self.input_tftensor[pdf_name])

            # construct and save full pdf spline
            full_pdf = pdf_prediction["full_pdf"].numpy()
            data = np.zeros(shape=(FLAV, POS_DENSITY, KPTS_H))
            xspace = np.linspace(XPOS_MIN, XPOS_MAX, POS_DENSITY)
            # fill out data tensor for spline
            for flav in range(FLAV):
                for n, _ in enumerate(xspace):
                    for i, _ in enumerate(KGRID2):
                        index = n*KPTS_H + i
                        data[flav][n][i] = full_pdf[0][index][flav]
            self.pdf_pred[pdf_name]["full_pdf"] = {}
            for n, flav_name in enumerate(flav_list[:FLAV]):
                self.pdf_pred[pdf_name]["full_pdf"][flav_name] = RectBivariateSpline(xspace, KGRID2, data[n], kx=3, ky=4)

            # construct and save collinear part spline
            coll_part = pdf_prediction["coll_part"].numpy()
            data = np.zeros(shape=(FLAV, POS_DENSITY))
            xspace = np.linspace(XPOS_MIN, XPOS_MAX, POS_DENSITY)
            # fill out data tensor for spline
            for flav in range(FLAV):
                for n, _ in enumerate(xspace):
                    index = n*KPTS_H
                    data[flav][n] = coll_part[0][index][flav]
            self.pdf_pred[pdf_name]["coll_part"] = {}
            for n, flav_name in enumerate(flav_list[:FLAV]):
                self.pdf_pred[pdf_name]["coll_part"][flav_name] = UnivariateSpline(xspace, data[n], k=5)

            # construct and save k dependence spline
            branch_k = pdf_prediction["branch_k"].numpy()
            data = np.zeros(shape=(FLAV, KPTS_H))
            # fill out data tensor for spline
            for flav in range(FLAV):
                for i, _ in enumerate(KGRID2):
                    index = i
                    data[flav][i] = branch_k[0][index][flav]
            self.pdf_pred[pdf_name]["branch_k"] = {}
            for n, flav_name in enumerate(flav_list[:FLAV]):
                self.pdf_pred[pdf_name]["branch_k"][flav_name] = UnivariateSpline(KGRID2, data[n], k=5)

        return loss_dict

    def get_pdf_list(self):
        return self.pdf_type

    def get_flav_list(self):
        return flav_list[0:FLAV]

    def get_full_pdf(self, pdf_name, flav_name, x, k2):
        return self.pdf_pred[pdf_name]["full_pdf"][flav_name].ev(x, k2)

    def get_coll_part(self, pdf_name, flav_name, x):
        return self.pdf_pred[pdf_name]["coll_part"][flav_name](x)

    def get_transverse_part(self, pdf_name, flav_name, k2):
        return self.pdf_pred[pdf_name]["branch_k"][flav_name](k2)

    def get_obs_pred(self):
        return self.obs_pred

    def get_target(self):
        return self.target

    def get_chi2pp_validation(self):
        # chi2 per point of the validation set
        return self.chi2pp_validation

    def get_chi2pp_kfold(self):
        # chi2 per point of the kfold set
        return self.chi2pp_kfold