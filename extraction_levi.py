import numpy as np
import matplotlib.pyplot as plt

class Pipe:
    def __init__(self, diameter, thickness, k, name: str = 'empty', material='undefined'):
        self.diameter = diameter
        self.thickness = thickness
        self.material = material
        self.name = name
        self.k = k
    def __str__(self):
        return(f'{self.name} is a pipe of diameter {self.diameter}, thickness {self.thickness}, made of {self.material} and with a thermal conductivity {self.k}')

    def area(self):
        d_i = (self.diameter - 2 * self.thickness) ** 2 * np.pi * 0.25
        return d_i
    
    
test = Pipe(diameter = 1, thickness = 0.1, k = 0.5)
print(test)
'''         
def conduction(T_h, Q, layers, Area): #this takes inputs in base SI units always
    layers = np.asarray(layers)
    L = layers[:, 0]
    k = layers[:, 1]

    R_t = np.sum(L / (k * Area))

    T_c = T_h - Q * R_t

    return T_c, R_t

q = ...
T_hot = ...
#first value is size/shape, second value is thermal conductivity
layers = [[3.35e-3, 401], #copper
          [2e-3, 3], #epoxy
          [5e-4, 237]] #aluminium 

#physical parameters of a coil
h = ...
l = ...
w = ...

A = h * l

T_cold = conduction(T_hot, q, layers, A)[0]

def pipe_convection(diameter, k_fluid, mu_fluid, m_dot, c_p, cooling = False, parallel = False):
    if cooling:
        n = 0.33
    else:
        n = 0.4
    RHS = 0.0233 * (m_dot * diameter / mu_fluid)**0.8 * (mu_fluid * c_p / k_fluid)**n
    h = k_fluid * RHS / diameter
    return h

A_contact = ... #area of contact between pipe and wall
T_diff_ww = q / (h * A_contact) #waterwall temperature difference

rho = ...
u = ...
A_cs_pipe = ...
massflow = u * A_cs_pipe * rho

def pipe_temp_diff(q, m_dot, c_p):
    return q / m_dot / c_p

'''