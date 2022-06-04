import tensorflow as tf
from tensorflow.keras.layers import multiply, Lambda
from tensorflow.keras import backend as K

import numpy as np

from phconst import e2, M
from config import FLAV, KPTS, KPTS2, KMAX, LWEIGHTS, POS_ALPHA, POS_DENSITY
from grid import WEIGHT_FUU, WEIGHT_FUT

from LagrangeMult import LagrangeMult

class Observable:
    def __init__(self):
        pass

class AUTsivers(Observable):
    """
        AUTsivers observable. The expected shape of the pdf outputs is (1, ndata*KPTS2, FLAV). The expected shape of the output (1, ndata) 
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.ndata = len(self.dataset["value"])

        # expected shape of the weights (1, ndata, KPTS2, 1) 
        self.weightFUU = np.repeat(WEIGHT_FUU, self.ndata, axis=1)
        self.weightFUT = np.repeat(WEIGHT_FUT, self.ndata, axis=1)

        for n, pT in enumerate(self.dataset["data"]["pT"]):
            for i in range(KPTS2):
                self.weightFUT[0][n][i][0] *= -pT/M

        # expected shape after next step (1, ndata, KPTS2, FLAV)
        self.weightFUU = np.repeat(self.weightFUU, FLAV, -1) * e2[:FLAV]
        self.weightFUT = np.repeat(self.weightFUT, FLAV, -1) * e2[:FLAV]

        super().__init__()

    def __call__(self, pdf_layer):
        upol_dist = tf.concat(tf.split(pdf_layer["upol_dist_" + self.dataset["target"]], num_or_size_splits=self.ndata, axis=-2), axis=0)
        upol_frag = tf.concat(tf.split(pdf_layer["upol_frag_" + self.dataset["hadron"]], num_or_size_splits=self.ndata, axis=-2), axis=0)
        sivers_dist = tf.concat(tf.split(pdf_layer["sivers_dist_" + self.dataset["target"]], num_or_size_splits=self.ndata, axis=-2), axis=0)

        # after next step the expected shape=(1, ndata, KPTS2, FLAV) 
        upol_dist = tf.expand_dims(upol_dist, axis=0)
        upol_frag = tf.expand_dims(upol_frag, axis=0)
        sivers_dist = tf.expand_dims(sivers_dist, axis=0)

        FUU = multiply([upol_dist, upol_frag, self.weightFUU])
        FUT = multiply([sivers_dist, upol_frag, self.weightFUT])

        # sum over flavours, expected shape=(1, ndata, KPTS2)
        FUU = K.sum(FUU, axis=-1)
        FUT = K.sum(FUT, axis=-1)

        # sum over momenta, expected shape=(1, ndata)
        FUU = K.sum(FUU, axis=-1)
        FUT = K.sum(FUT, axis=-1)

        y_pred = FUT/FUU

        # output shape=(1, ndata)
        return y_pred

class POSObservable(Observable):
    """
        Positivity observable. Calculates positivity characteristics of the pdfs' outputs and assigns some weights which depend on
        whether a positivity condition is satisfied
        The expected shape of the pdf output is (1, ndata*KPTS_H, FLAV). The expected shape of the output (1, 1) 
    """
    def __init__(self, pdf_type, ndata, pos_type="ALL", alpha=POS_ALPHA, LWeight=LWEIGHTS, **kwargs):
        self.pdf_type = pdf_type
        self.ndata = ndata
        self.alpha = alpha
        self.LWeight = LWeight
        self.pos_type=pos_type
        self.LMlayer = LagrangeMult(FLAV, name="LMLayer")
        super().__init__(**kwargs)

    def __call__(self, pdf_layer):
        y_pred = []
        for pdf_name in self.pdf_type:
            # split the pdf output into momenta and x dependence
            full_output = tf.split(pdf_layer[pdf_name]["full_pdf"], num_or_size_splits=self.ndata, axis=-2)
            full_output = tf.concat(full_output, axis=0)
            full_output = tf.expand_dims(full_output, axis=0)
            # at this point the expected shape of the full_output is (1, ndata, KPTS_H, FLAV)

            # split the momentum part of the pdf
            k_output = tf.split(pdf_layer[pdf_name]["branch_k"], num_or_size_splits=self.ndata, axis=-2)
            k_output = tf.concat(k_output, axis=0)
            k_output = tf.expand_dims(k_output, axis=0)

            # the pdf function should decrease in k space. Calculate a difference between adjasent points in k space
            k_output_diff = k_output[:,:,:-1,:] - k_output[:,:,1:,:]

            # the weights are calculated with the elu function
            fun = getattr(K, "elu")
            full_output = fun(-full_output, alpha=self.alpha)
            k_output = fun(-k_output, alpha=self.alpha)
            k_output_diff = fun(-k_output_diff, alpha=self.alpha)

            # sum over momenta, the expected shape is (1, ndata, FLAV)
            full_output = K.sum(full_output, axis=-2)
            k_output = K.sum(k_output, axis=-2)
            k_output_diff = K.sum(k_output_diff, axis=-2)

            # apply LambdaMultipliers
            full_output = self.LMlayer(full_output)
            k_output = self.LMlayer(k_output)
            k_output_diff = self.LMlayer(k_output_diff)

            # sum over flavours, the expected shape is (1, ndata)
            full_output = K.sum(full_output, axis=-1)
            k_output = K.sum(k_output, axis=-1)
            k_output_diff = K.sum(k_output_diff, axis=-1)

            # expected shape after this step is (1, ndata, 1)
            full_output = tf.expand_dims(full_output, axis=-1)
            k_output = tf.expand_dims(k_output, axis=-1)
            k_output_diff = tf.expand_dims(k_output_diff, axis=-1)

            # total weight is defferent depending on whether the pdf output should be positive,
            # decrease of both positive and decrease
            if self.pos_type=="FPOS":
                diff = full_output + k_output
            elif self.pos_type=="FDEC":
                diff = k_output_diff
            else:
                diff = full_output + k_output + k_output_diff

            y_pred.append(diff)

        # concatenate pdf outputs, expected shape after this (1, ndata, total_pdf_num)
        y_pred = tf.concat(y_pred, axis=-1)
        # sum over all pdfs, expected shape after this step (1, ndata)
        y_pred = K.sum(y_pred, axis=-1)

        # final shape=(1, POS_DENSITY)
        return y_pred