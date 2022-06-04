import numpy as np
from scipy.interpolate import RectBivariateSpline
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras import backend as K

from phconst import flav as active_flav
from phconst import Q20 as Q2init

class UpolDistLayer():
    def __init__(self, pdf_type):
        self.fmask = ['bb', 'cb', 'sb', 'ub', 'db', 'd', 'u', 's', 'c', 'b', 'g']

        if "proton" in pdf_type:
            f = open('./data/DGLAP/CJ15lo_0000.dat', 'r')
            self.file = f.readlines()
            f.close()
        if "pi" in pdf_type:
            f = open('./data/DGLAP/dsspipLO_0000.dat', 'r')
            self.file = f.readlines()
            f.close()

        self.x = np.array([float(n) for n in self.file[3].split()])
        self.Q = np.array([float(n) for n in self.file[4].split()])
        self.flav = self.file[5].split()
        self.data = np.zeros((len(self.flav), len(self.Q), len(self.x)))

        for flav in range(len(self.flav)):
            for Q in range(len(self.Q)):
                for x in range(len(self.x)):
                    self.data[flav][Q][x] = float(self.file[6 + Q + x * len(self.Q)].split()[flav])

        self.interpolator = {}
        cnt = 0
        for flav in self.fmask:
            self.interpolator[flav] = RectBivariateSpline(self.x,self.Q,self.data[cnt].T,kx=3, ky=4)
            cnt+=1

        def collinear_processor(tensor):
            tensor = tf.concat([self.get_collinear(flav, tensor, Q2init) for flav in active_flav], axis=-1)
            return tf.cast(tensor, dtype="float32")

        self.collinear_layer = collinear_processor

    def get_collinear(self, flav, x, Q2):
        return self.interpolator[flav].ev(x, np.sqrt(Q2)) / x

    def __call__(self, tensor):
        return self.collinear_layer(tensor)

if __name__=="__main__":
    dist_layer = UpolDistLayer("upol_dist_proton")

    tensor = K.constant([[[0.5], [0.5], [0.5], [0.5], [0.5]]])
    result = dist_layer(tensor)

    print("Done")