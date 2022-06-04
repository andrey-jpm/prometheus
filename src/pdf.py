import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from config import KPTS, KMAX, KPTS2, FLAV
from upoldist import UpolDistLayer

class DenseLayer():
    def __init__(self, nodes, activations, initializer):
        self.list_of_layers = []
        nod_prev = 1
        for nod_cur, act in zip(nodes, activations):
            self.list_of_layers += [Dense(units=int(nod_cur), activation=act, input_shape=(nod_prev,), kernel_initializer=initializer)]
            nod_prev = int(nod_cur)

    def __call__(self, input_tensor):
        tensor = self.list_of_layers[0](input_tensor)
        for layer in self.list_of_layers[1:]:
            tensor = layer(tensor)

        return tensor

class EvolvPDF():
    def __init__(self):
        # constants to modify x dependence
        self.alpha = np.random.rand()
        self.beta = np.random.rand()
        # constants to modify k dependence
        self.width_k2 = (1-0.3)*np.random.rand() + 0.3

    def evolve_X(self, branch_x):
        output = tf.math.pow(branch_x, 1-self.alpha)
        output *= tf.math.pow(1-branch_x, self.beta)
        return output

    def evolve_K(self, branch_k):
        output = tf.math.exp(-branch_k/self.width_k2) / (np.pi * self.width_k2)
        return tf.cast(output, dtype="float32")


def _get_norm_input(pdf_type):
    # generate numpy input for normalization of the network which simulates k dependence
    norm_input = np.zeros(shape=(1,KPTS2,1))
    # generate weights for normalization integration
    weights = np.ones(shape=(1,KPTS2,1))
    kval = np.linspace(-KMAX, KMAX, KPTS)
    A = (2*KMAX/(KPTS-1))**2
    for i, kx in enumerate(kval):
        for j, ky in enumerate(kval):
            index = i*KPTS + j
            norm_input[0][index][0] = kx**2 + ky**2
            if (i == 0 and j == 0) or (i == 0 and j == KPTS - 1) or (i == KPTS - 1 and j == 0) or (i == KPTS - 1 and j == KPTS - 1):
                weights[0][index][0] = A/4
            elif (i == 0) or (i == KPTS - 1) or (j == 0) or (j == KPTS - 1):
                weights[0][index][0] = A/2
            else:
                weights[0][index][0] = A

    weights = np.repeat(weights, FLAV, axis=-1)
    weights = K.constant(weights)

    # generate tensors for normalization input
    norm_tftensor = K.constant(norm_input)
    norm_ktensor = Input(shape=norm_input.shape[1:], batch_size=1, name = "norm_inp_" + pdf_type)

    return norm_ktensor, norm_tftensor, weights

def get_pdf_nn(nodes, activations, initializer, pdf_type):
    # generate placeholder input
    placeholder_input = Input(shape=(None,2),batch_size=1, name="pdf_inp_" + pdf_type)

    # generate input tensors of the model
    input_tftensor = {placeholder_input.name: None}
    input_ktensor = {placeholder_input.name: placeholder_input}

    # split input tensor into x and k tensors
    split_layer = Lambda(tf.split, arguments={"num_or_size_splits":2, "axis":-1})

    # don't model x dependence of unpolarized pdfs, we'll use collinear input instead
    if "upol" not in pdf_type:
        dencel_x = DenseLayer(nodes, activations, initializer)
    else:
        dencel_x = None

    dencel_k = DenseLayer(nodes, activations, initializer)

    norm_ktensor, norm_tftensor, weights = _get_norm_input(pdf_type)

    # add normalization inputs to input dictionaries
    input_tftensor[norm_ktensor.name] = norm_tftensor
    input_ktensor[norm_ktensor.name] = norm_ktensor

    # add auxiliary input we will use to predict pdf value
    auxinp_ktensor = Input(shape=(None,2), batch_size=1, name="aux_inp_" + pdf_type)

    input_tftensor[auxinp_ktensor.name] = None
    input_ktensor[auxinp_ktensor.name] = auxinp_ktensor

    evol_layer = EvolvPDF()

    def full_pdf(tensor):
        # split the input into two branches
        branch_x, branch_k = split_layer(tensor)

        # evolution coefficients to speed up training
        evol_x = evol_layer.evolve_X(branch_x)
        evol_k = evol_layer.evolve_K(branch_k)
        evol_norm = evol_layer.evolve_K(norm_ktensor)

        # dence the k input which simulates momentum dependence of the network
        branch_k = dencel_k(branch_k)*tf.repeat(evol_k,FLAV,axis=-1)

        # calculation of the normalization factor
        norm = dencel_k(norm_ktensor)*tf.repeat(evol_norm,FLAV,axis=-1)
        norm = weights*norm
        norm = K.sum(norm, axis=-2)
        norm = tf.reshape(norm, (-1,))
        norm = 1/norm

        # normalize the output
        branch_k = norm * branch_k

        # if pdf is polarized dense x input as well
        if "upol" not in pdf_type:
            branch_x = evol_x*dencel_x(branch_x)
            return {"branch_x": branch_x, "branch_k": branch_k}
        else:
            return {"branch_k": branch_k}

    # generate the model
    pdf_model = Model(input_ktensor, {"pdf_output": full_pdf(placeholder_input), "aux_output": full_pdf(auxinp_ktensor)},
        name="pdf_model_"+pdf_type)
    
    return pdf_model, input_ktensor, input_tftensor

class FullPDFModel():
    def __init__(self, pdf_model, pdf_type, input_tensor):
        # full input tensor
        self.input_tensor = input_tensor
        # pdf and aux inputs corresponding to experimental points
        self.pdf_input = self.input_tensor["pdf_inp_" + pdf_type]
        self.aux_input = self.input_tensor["aux_inp_" + pdf_type]

        # pdf model and its type
        self.pdf_model = pdf_model
        self.pdf_type = pdf_type

        # generate collinear parts
        self.pdf_x_input, self.pdf_col_part = self._build_collinear(self.pdf_input, self.pdf_type)
        self.aux_x_input, self.aux_col_part = self._build_collinear(self.aux_input, self.pdf_type)

    def __call__(self, input_tensor):
        pdf_output = self.pdf_model(input_tensor)["pdf_output"]
        # use the collinear part and pdf_output of the network
        if "branch_x" in pdf_output.keys():
            return self.pdf_col_part*pdf_output["branch_x"]*pdf_output["branch_k"]
        else:
            return self.pdf_col_part*pdf_output["branch_k"]

    def _build_collinear(self, pdf_input, pdf_type):
        x_input = tf.split(pdf_input, num_or_size_splits=2, axis=-1)[0].numpy()

        if "upol" in pdf_type:
            col_part = UpolDistLayer(pdf_type)(x_input)
        else:
            shape = tuple(list(x_input.shape[:-1]) + [FLAV])
            col_part = K.constant(np.ones(shape=shape))

        return x_input, col_part

    def predict(self, input_tensor):
        # calculate pdf output on "aux_output" which is equal to the positivity grid
        aux_output = self.pdf_model(input_tensor)["aux_output"]
        if "branch_x" in aux_output.keys():
            return {"full_pdf": self.aux_col_part*aux_output["branch_x"]*aux_output["branch_k"],
                    "coll_part": self.aux_col_part*aux_output["branch_x"],
                    "branch_k": aux_output["branch_k"]}
        else:
            return {"full_pdf": self.aux_col_part*aux_output["branch_k"],
                    "coll_part": self.aux_col_part,
                    "branch_k": aux_output["branch_k"]}

        