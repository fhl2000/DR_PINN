import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from module.data_gen import ReferenceU_vc,CompoundGeometry,databuilderforAll
from module.optimizers import lm_train,gdlm_train, adam_train, lbfgs_train
from module.model import Shallow_ext,Deep_ext,Shallow_Deep_ext
from module.compute_loss import *
from module.visualize import *
from utils import *
from setting import *
from loss_fun import *
import time

def gen_train_dic(data, full_batch=True, device="cpu"):
    # use fixed sampling points
    
    x_dict,data_dict=data

    X_bd=x_dict["boundary"].to(torch.float64).to(device)
    U_bd=data_dict["boundary"]["g_D"].to(torch.float64).to(device)

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

    # only for adam or lbfgs
    weight_bd = 100
    weight_f = 1

    if full_batch:
        # full-batch mode
        while True:
            train_dic={}
            train_dic["bd"]= (loss_boundary_coupling,(X_bd,U_bd, weight_bd))
            train_dic["f"]= (loss_f_coupling,(X_in,U_f, Sign, Beta, Gradbeta, weight_f))
            train_dic["jump"]= (loss_jump_coupling,(X_interf, U_w))
            train_dic["normal_jump"]= (loss_normal_jump_coupling,(X_interf, Normal_ij,U_v,iBeta1,iBeta2))
            yield train_dic

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

def main(args):
    ###### configs #######
    if_train=args.if_train
    continue_train=args.continue_train
    network_type=args.network
    if args.optimizer==0:
        optimizer_type="gdlm"
    elif args.optimizer==1:
        optimizer_type="lm"
    elif args.optimizer==2:
        optimizer_type="adam"
    elif args.optimizer==3:
        optimizer_type="adam_lbfgs"
    model_path=f"data_cache/comparison_{information}_{network_type}_{optimizer_type}_d_{args.seed}.pt"


    ######################
    if os.path.isfile(model_path):
        checkpoint=torch.load(model_path)

    device = torch.device('cuda:0')
    torch.set_default_dtype(torch.float64)

    # get geometry
    def get_geometry():        
        g_base= CompoundGeometry(geoIn=geo1, geoOut=geo2)
        return g_base

    g_base = get_geometry()

    def beta1(x):
        return torch.ones(x.shape[0]).to(x.device) 
    def beta2(x):
        return torch.ones(x.shape[0]).to(x.device)

    reference_u = ReferenceU_vc(g_base,u1=u1,u2=u2, beta1=beta1, beta2=beta2)

    if network_type=="Modified":
        model= Shallow_Deep_ext(3, 20, 2, layers=4).to(device)
    elif network_type=="Deep":
        model= Deep_ext(3, 20, 2, layers=4).to(device)
    elif network_type=="Shallow":
        model= Shallow_ext(3, 250, 2).to(device)
    print("model:\n",model)
    func_params= dict(model.named_parameters())


    data=databuilderforAll(g_base,reference_u, random_method="Hammersley", device=device,
                            n_inner=n_inner,n_boundary=n_bondary,n_interface=n_interface)

    time1=time.time()
    if optimizer_type=="gdlm":
        train_dic=prepare_train_All(model,func_params,data)
        func_params = gdlm_train(train_dic,LM_iter=2000,imethod=2,iaccel=1,ibold=2,continue_train=continue_train,model_path=model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1})
        wholenet = lambda x: functional_call(model,func_params,x) 
    elif optimizer_type=="lm":
        train_dic=prepare_train_All(model,func_params,data)
        func_params = lm_train(train_dic,LM_iter=6000,continue_train=continue_train,model_path=model_path)
        wholenet = lambda x: functional_call(model,func_params,x) 
    elif optimizer_type=="adam":
        generator=gen_train_dic(data, full_batch=True, device=device)
        adam_train(generator, model, epoch= 1000, learning_rate=0.001, model_path=model_path)
        wholenet = model
    elif optimizer_type=="adam_lbfgs":
        generator=gen_train_dic(data, full_batch=True, device=device)
        adam_train(generator, model, epoch= 20, learning_rate=0.001, model_path=model_path)
        lbfgs_train(generator, model, max_iter=20000, continue_training=True, model_path=model_path)
        wholenet = model
    time2=time.time()
    total_time=time2-time1
    
    eval_3d_composed(wholenet,g_base, reference_u, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--if_test', action="store_true")
    parser.add_argument('--if_train', type=bool, default=True)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--network', type=str, default="Modified")  # Modified, Deep, Shallow
    parser.add_argument('--optimizer', type=int, default=0)  # 0 for gdlm, 1 for lm, 2 for adam 3 for adam + lbfgs
    args = parser.parse_args()

    if args.if_test:
        args.if_train = False
        
    setup_seed(args.seed)
    main(args)

