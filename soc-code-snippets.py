######################################

import numpy as np
import random

def generate():
    Ux = random.random()*2-1
    Uy = random.random()*2-1
    Vx = random.random()*2-1

    s = np.sqrt(4*(Ux**2)+(Uy+Vx)**2)/2
    ω = (Vx-Uy)/2

    if np.abs(s) <= np.abs(ω):
        ωRES = np.sign(ω)*(np.abs(ω)-np.abs(s))
    else:
        ωRES = 0

    return Ux,Uy,Vx,ωRES

######################################