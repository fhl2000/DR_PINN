import torch
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from module import geometry as geo

u1 = lambda x: torch.exp(x[:,0]**2+x[:,1]**2+x[:,2]**2)
u2 = lambda x: 0.1*(x[:,0]**2+x[:,1]**2+x[:,2]**2)**2-0.01*torch.log(2 *torch.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2))


geo1 = geo.Ellipsoid([0,0,0],[0.7,0.5,0.3])
geo2 = geo.Cuboid([-1,-1,-1],[1,1,1])

information = "3D_ellipsoid"

n_interface = 800
n_inner = 15000
n_bondary = 2400

# n_interface = 800
# n_inner = 9000
# n_bondary = 2400


# n_interface = 800
# n_inner = 6000
# n_bondary = 800

device = torch.device("cuda:0")
dim=3



def phi1(x):
    return (x[0]/0.7)**2+(x[1]/0.5)**2+(x[2]/0.3)**2-1



