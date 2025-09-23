import numpy as np
import matplotlib.pyplot as plt

def coldtemp(T_h, Q, layers, Area):
    L = layers[:, 0]
    k = layers[:, 1]

    R_t = np.sum(L / (k * Area))

    T_c = T_h - Q * R_t

    return T_c, R_t


#change these values (all in SI UNits):
heat_in = 4000
hot_temp = 80
layers = np.array([[1, 10] #First layer (first value is length, second value is conductivity)
                   [2, 15]
                   [3, 30]])

Area = 40 #in m^2

#provides answer
print(coldtemp(hot_temp, heat_in, layers, Area))