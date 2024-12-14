import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from module import geometry as geo
from module.data_gen import ReferenceU,CompoundGeometry,databuilderforPatch,databuilderforBase
from module.optimizers import lm_train,gdlm_train
from module.model import Shallow_ext,Shallow_Deep_ext
from module.compute_loss import *
from module.visualize import *
from module.utils import seed_all,patch_laplace
from module.FD_solver import FFT_poisson2D_fast_solver

import numpy as np
import torch

###### configs #######
seed=0
if_train_p=True
continue_train_p=False
patch_model_path="./data_cache/circle_a_patch.pt"
visualize_patchnet=False
if_train_r=True
continue_train_r=False
raw_model_path="./data_cache/cirlce_a_raw.pt"
######################
if os.path.isfile(patch_model_path):
    checkpoint=torch.load(patch_model_path)
if os.path.isfile(raw_model_path):
    checkpoint1=torch.load(raw_model_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
seed_all(seed)

u1 = lambda x: torch.exp(x[:,0]**2+x[:,1]**2)
u2 = lambda x: 0.1*(x[:,0]**2+x[:,1]**2)**2-0.01*torch.log(2 *torch.sqrt(x[:,0]**2+x[:,1]**2))

# get geometry
def get_geometry():
    geo1 = geo.Disk([0,0],1/2)
    geo2 = geo.Rectangle([-1,-1],[1,1])
    g_base= CompoundGeometry(geoIn=geo1, geoOut=geo2)
    return g_base

def prepare_train_patch(model,func_params, patchdata):
    # set up ._func_model for compute_loss functions
    setup_fm(model)
    x_dict,data_dict=patchdata
    X_bd=X_ij=X_fj=x_dict["interface"].to(torch.float64)
    U_bd=data_dict["interface"]["w"].to(torch.float64)
    Normal_ij=data_dict["interface"]["interface_normal_vector"].to(torch.float64)
    Unj_ij=data_dict["interface"]["v"].to(torch.float64)
    Ufj=data_dict["interface"]["f_jump"].to(torch.float64)

    train_dic={"bd":(compute_loss_bd, (None, 0, 0), (func_params, X_bd, U_bd)),
               "normal_jump":(compute_loss_normal_jump, (None, 0, 0, 0), (func_params, X_ij, Normal_ij, Unj_ij)),
               "fj":(compute_loss_fj, (None, 0, 0), (func_params, X_fj, Ufj)),}
    return train_dic

g_base = get_geometry()
reference_u = ReferenceU(g_base,u1=u1,u2=u2)


model_p = Shallow_ext(2, 30, 1).to(device)
print("patch model:\n",model_p)
# func_model_p, func_params_p = make_functional(model_p)
func_params_p= dict(model_p.named_parameters())
model_p.to("meta")

if if_train_p:
    patchdata=databuilderforPatch(g_base,reference_u,n_interface=120,device=device)
    # plot_samples2d(patchdata[0])
    if continue_train_p:
        print(f"loaded patch model from {patch_model_path}")
        func_params_p=checkpoint["func_params"]
    train_dic=prepare_train_patch(model_p,func_params_p,patchdata)
    # lm_train(train_dic,continue_train=continue_train,model_path=patch_model_path)
    func_params_p = gdlm_train(train_dic,LM_iter=2000,imethod=2,iaccel=1,ibold=2,continue_train=continue_train_p,model_path=patch_model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1}) 

else:
    func_params_p=checkpoint["func_params"]
    print('Last trained patchnet from %s' % (patch_model_path,) )

patchnet = lambda x: functional_call(model_p,func_params_p,x)

if visualize_patchnet:
    x=g_base.uniform_sampler(8000)[1].astype(np.float64)
    v=patchnet(torch.tensor(x)).detach().cpu().numpy()
    plot_results3d(x,v,"patchnet")



######Stage 2#######

def phi(x):
    return x[0]**2+x[1]**2 -1/4

def af1(x):
    p=phi(x)
    return torch.abs(p)*p # phi2

def prepare_train_r(model,func_params, data):
    # set up ._func_model for compute_loss functions
    setup_fm(model)
    x_dict,data_dict=data

    X_bd=x_dict["boundary"].to(torch.float64)
    U_bd=data_dict["boundary"]["g_D"].to(torch.float64)

    X_in=x_dict["inner"].to(torch.float64)
    U_f=data_dict["inner"]["f"].to(torch.float64)

    train_dic={"bd":(compute_loss_bd, (None, 0, 0), (func_params, X_bd, U_bd)),
               "f":(compute_loss_f, (None, 0, 0), (func_params, X_in, U_f)),}
    return train_dic

model_r = Shallow_Deep_ext(2, 20, 1, layers=2,addiction_features=[af1]).to(device)


func_params_r= dict(model_r.named_parameters())
model_r.to("meta")

if if_train_r:
    data_r = databuilderforBase(g_base,patchnet,reference_u,random_method="Hammersley",device=device)
    # plot_samples2d(data_r[0])
    if continue_train_r:
        print(f"loaded raw model from {raw_model_path}")
        func_params_r=checkpoint1["func_params"]
    train_dic=prepare_train_r(model_r,func_params_r,data_r)
    # lm_train(train_dic,continue_train=continue_train_r,model_path=raw_model_path)
    func_params_r = gdlm_train(train_dic,imethod=2,iaccel=1,ibold=2,continue_train=continue_train_r,model_path=raw_model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1}) 

else:
    try:
        func_params_r=checkpoint1["func_params"]
    except Exception:
        exit()
    print('Last trained rawnet from %s' % (raw_model_path,) )


rawnet = lambda x: vmap(functional_call,(None,None,0))(model_r,func_params_r,x)
eval_2d(rawnet,patchnet,g_base,reference_u,device=device)

# FD fast solver for stage 2

def boundary(x: np.ndarray):
    return reference_u.u2(torch.tensor(x)).numpy()
def f(x:np.ndarray):
    ans=reference_u.f(torch.tensor(x)).numpy()
    patch_index= g_base.inside_u1(x)
    ans[patch_index] += patch_laplace(torch.tensor(x[patch_index],device=device),patchnet, device=device).detach().cpu().numpy()
    return ans

resolution = 1024 
U_pred = FFT_poisson2D_fast_solver(f,boundary,resolution)
eval_2d_FD(U_pred,patchnet,g_base,reference_u,resolution,device=device)