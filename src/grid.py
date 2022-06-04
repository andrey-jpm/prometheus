import numpy as np
from config import KMAX, KMAX2, KPTS, KPTS2, KPTS_H, XPOS_MIN, XPOS_MAX, POS_DENSITY

# momentum grids
KGRID = np.linspace(-KMAX, KMAX, KPTS)
KGRID2 = np.linspace(0, KMAX2, KPTS_H)

# weights for integration
WEIGHT_FUU = np.ones(shape=(1,KPTS2,1))
WEIGHT_FUT = np.ones(shape=(1,KPTS2,1))
A = (2*KMAX/(KPTS-1))**2
for i, kx in enumerate(KGRID):
    for j, ky in enumerate(KGRID):
        index = i*KPTS + j
        if (i == 0 and j == 0) or (i == 0 and j == KPTS - 1) or (i == KPTS - 1 and j == 0) or (i == KPTS - 1 and j == KPTS - 1):
            WEIGHT_FUU[0][index][0] = A/4
        elif (i == 0) or (i == KPTS - 1) or (j == 0) or (j == KPTS - 1):
            WEIGHT_FUU[0][index][0] = A/2
        else:
            WEIGHT_FUU[0][index][0] = A

        WEIGHT_FUT[0][index][0] = WEIGHT_FUU[0][index][0] * kx
        
WEIGHT_FUU = np.expand_dims(WEIGHT_FUU, axis=0)
WEIGHT_FUT = np.expand_dims(WEIGHT_FUT, axis=0)

# grid for positibity input and auxiliary pdf output
POS_SIZE = POS_DENSITY*KPTS_H
POS_INPUT_GRID = np.zeros(shape=(1,POS_SIZE,2))
# for positivity dataset generate a pseudo distribution in x
xspace = np.linspace(XPOS_MIN, XPOS_MAX, POS_DENSITY)
for n, xB in enumerate(xspace):
    for i, k2 in enumerate(KGRID2):
        index = n*KPTS_H + i
        POS_INPUT_GRID[0][index][0] = xB
        POS_INPUT_GRID[0][index][1] = k2