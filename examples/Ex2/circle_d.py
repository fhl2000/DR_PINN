import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from module import geometry as geo
from module.data_gen import ReferenceU_vc,CompoundGeometry,databuilderforAll
from module.optimizers import lm_train,gdlm_train
from module.model import Shallow_Deep_ext
from module.compute_loss import *
from module.visualize import *
from module.utils import seed_all

import numpy as np 
import torch

###### configs #######
seed=0
if_train=True
continue_train=False
model_path="./data_cache/circle_d.pt"


######################
if os.path.isfile(model_path):
    checkpoint=torch.load(model_path)

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
seed_all(seed)

u1 = lambda x: torch.exp(x[:,0]**2+x[:,1]**2)
u2 = lambda x: 0.1*(x[:,0]**2+x[:,1]**2)**2-0.01*torch.log(2 *torch.sqrt(x[:,0]**2+x[:,1]**2))
# beta1= lambda x: 1.0*torch.ones(len(x))
# beta2= lambda x: 1.0*torch.ones(len(x))
# get geometry
def get_geometry():
    geo1 = geo.Disk([0,0],1/2)
    geo2 = geo.Rectangle([-1,-1],[1,1])
    
    g_base= CompoundGeometry(geoIn=geo1, geoOut=geo2)
    return g_base

def prepare_train_All(model, func_params, data):
    # set up ._func_model for compute_loss functions
    setup_fm(model)

    x_dict,data_dict=data

    X_bd=x_dict["boundary"].to(torch.float64)
    U_bd=data_dict["boundary"]["g_D"].to(torch.float64)

    X_in=x_dict["inner"].to(torch.float64)
    U_f=data_dict["inner"]["f"].to(torch.float64)
    Sign=data_dict["inner"]["sign"].to(torch.float64)
    Beta= data_dict["inner"]["beta"].to(torch.float64)
    Gradbeta= data_dict["inner"]["gradbeta"].to(torch.float64)

    X_interf=x_dict["interface"].to(torch.float64)
    U_w=data_dict["interface"]["w"].to(torch.float64)
    Normal_ij=data_dict["interface"]["normal_vector"].to(torch.float64)
    U_v=data_dict["interface"]["v"].to(torch.float64)
    iBeta1=data_dict["interface"]["ibeta1"].to(torch.float64)
    iBeta2=data_dict["interface"]["ibeta2"].to(torch.float64)
    

    train_dic={"bd":(compute_loss_bd_composed, (None, 0, 0), (func_params, X_bd, U_bd)),
               "jump": (compute_loss_jump_composed, (None, 0, 0), (func_params, X_interf, U_w)),
               "normal_jump":(compute_loss_normal_jump_composed, (None, 0, 0, 0, 0, 0), (func_params, X_interf, Normal_ij, U_v, iBeta1, iBeta2)),
               "f":(compute_loss_f_composed, (None, 0, 0, 0, 0, 0), (func_params, X_in, U_f, Sign, Beta, Gradbeta)),
    }

    return train_dic

g_base = get_geometry()
reference_u = ReferenceU_vc(g_base,u1=u1,u2=u2)

model= Shallow_Deep_ext(2, 25, 2, layers=2).to(device)   # output[0] is U,  output[1] is V
print("model:\n",model)
func_params= dict(model.named_parameters())
model.to("meta")

if if_train:
    data=databuilderforAll(g_base,reference_u, random_method="Hammersley",device=device)
    # plot_samples2d(data[0])
    if continue_train:
        print(f"loaded patch model from {model_path}")
        func_params=checkpoint["func_params"]
    train_dic=prepare_train_All(model,func_params,data)
    # lm_train(train_dic,continue_train=continue_train,model_path=patch_model_path)
    func_params = gdlm_train(train_dic,LM_iter=2000,imethod=2,iaccel=1,ibold=2,continue_train=continue_train,model_path=model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1}) 

else:
    func_params=checkpoint["func_params"]
    print('Last trained patchnet from %s' % (model_path,) )

wholenet = lambda x: functional_call(model,func_params,x)
eval_2d_composed(wholenet,g_base, reference_u,device=device)


