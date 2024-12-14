import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from module.data_gen import ReferenceU_vc,CompoundGeometry,databuilderforPatch,databuilderforBaseVC
from module.optimizers import lm_train,gdlm_train,adam_train,lbfgs_train
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
    Beta= data_dict["inner"]["beta"].to(torch.float64)
    Gradbeta= data_dict["inner"]["gradbeta"].to(torch.float64)
    Sign= data_dict["inner"]["sign"].to(torch.float64)

    X_interf=x_dict["interface"].to(torch.float64)
    Normal_ij=data_dict["interface"]["normal_vector"].to(torch.float64)
    U_v=data_dict["interface"]["normal_jump"].to(torch.float64)
    iBeta1=data_dict["interface"]["ibeta1"].to(torch.float64)
    iBeta2=data_dict["interface"]["ibeta2"].to(torch.float64)

    # weights for adam or lbfgs
    weight_bd = 100
    weight_f = 1



    if full_batch:
        # full-batch mode
        while True:
            train_dic={}
            train_dic["bd"]= (loss_boundary_decoupling,(X_bd,U_bd, weight_bd))
            train_dic["f"]= (loss_f_decoupling,(X_in,U_f, Beta, Gradbeta, weight_f))
            train_dic["normal_jump"]= (loss_normal_jump_decoupling,(X_interf, Normal_ij, U_v, iBeta1, iBeta2))
            yield train_dic

def prepare_train_patch(model,func_params, patchdata):
    # set up ._func_model for compute_loss functions
    setup_fm(model)

    x_dict,data_dict=patchdata
    X_bd=X_ij=X_fj=x_dict["interface"].to(torch.float64)
    U_bd=data_dict["interface"]["w"].to(torch.float64)
    

    train_dic={"bd":(compute_loss_bd, (None, 0, 0), (func_params, X_bd, U_bd)),
            }
    return train_dic

def prepare_train_r(model,func_params, data):
    # set up ._func_model for compute_loss functions
    setup_fm(model)

    x_dict,data_dict=data

    X_bd=x_dict["boundary"].to(torch.float64)
    U_bd=data_dict["boundary"]["g_D"].to(torch.float64)

    X_in=x_dict["inner"].to(torch.float64)
    U_f=data_dict["inner"]["f"].to(torch.float64)
    Beta=data_dict["inner"]["beta"].to(torch.float64)
    Gradbeta=data_dict["inner"]["gradbeta"].to(torch.float64)

    X_interf=x_dict["interface"].to(torch.float64)
    U_inj=data_dict["interface"]["normal_jump"].to(torch.float64)
    Normal_ij=data_dict["interface"]["normal_vector"].to(torch.float64)
    Ibeta1=data_dict["interface"]["ibeta1"].to(torch.float64)
    Ibeta2=data_dict["interface"]["ibeta2"].to(torch.float64)

    train_dic={"bd":(compute_loss_bd, (None, 0, 0), (func_params, X_bd, U_bd)),
            "f":(compute_loss_f_vc, (None, 0, 0, 0, 0), (func_params, X_in, U_f, Beta, Gradbeta)),
            "inj":(compute_loss_inj_vc, (None, 0, 0, 0, 0, 0),(func_params, X_interf, U_inj, Normal_ij, Ibeta1, Ibeta2)),
    }
    return train_dic

def main(args):
    ###### configs #######
    network_type=args.network
    if args.optimizer==0:
        optimizer_type="gdlm"
    elif args.optimizer==1:
        optimizer_type="lm"
    elif args.optimizer==2:
        optimizer_type="adam"
    elif args.optimizer==3:
        optimizer_type="adam_lbfgs"
    continue_train=args.continue_train
    patch_model_path=f"data_cache/comparison_{information}_{network_type}_{optimizer_type}_c_patch_{args.seed}.pt"

    continue_train_r=args.continue_train
    raw_model_path=f"data_cache/comparison_{information}_{network_type}_{optimizer_type}_c_raw_{args.seed}.pt"
    ######################
    if os.path.isfile(patch_model_path):
        checkpoint=torch.load(patch_model_path)
    if os.path.isfile(raw_model_path):
        checkpoint1=torch.load(raw_model_path)


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


    model_p = Shallow_Deep_ext(3, 20, 1, 2).to(device)
    print("patch model:\n",model_p)
    func_params_p= dict(model_p.named_parameters())
    model_p.to("meta")


    patchdata=databuilderforPatch(g_base,reference_u,n_interface=n_interface, device=device)
    if continue_train:
        print(f"loaded patch model from {patch_model_path}")
        func_params_p=checkpoint["func_params"]
    train_dic=prepare_train_patch(model_p,func_params_p,patchdata)
    func_params_p = gdlm_train(train_dic,LM_iter=800,imethod=2,iaccel=1,ibold=2,continue_train=continue_train,model_path=patch_model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1}) 

    patchnet = lambda x: functional_call(model_p,func_params_p,x)


    ######Stage 2#######


    compute_loss_inj_vc._phi=phi1
    def af1(x):
        phi=phi1(x)
        return torch.abs(phi)
    
    parallel_af1 = lambda x: vmap(af1,(0,))(x)


    if network_type=="Modified":
        if args.optimizer<=1:
            model_r = Shallow_Deep_ext(3, 20, 1, layers=4,addiction_features=[af1]).to(device)
        else:
            model_r = Shallow_Deep_ext(3, 20, 1, layers=4,addiction_features=[parallel_af1]).to(device)
    elif network_type=="Deep":
        if args.optimizer<=1:
            model_r = Deep_ext(3, 20, 1, layers=4,addiction_features=[af1]).to(device)
        else:
            model_r = Deep_ext(3, 20, 1, layers=4,addiction_features=[parallel_af1]).to(device)
    elif network_type=="Shallow":
        if args.optimizer<=1:
            model_r = Shallow_ext(3, 250, 1,addiction_features=[af1]).to(device)
        else:
            model_r = Shallow_ext(3, 250, 1,addiction_features=[parallel_af1]).to(device)
    print("model:\n",model_r)
    
    func_params_r= dict(model_r.named_parameters())


    data_r = databuilderforBaseVC(g_base,patchnet,reference_u,random_method="Hammersley", device=device,
                                    n_inner=n_inner,n_boundary=n_bondary,n_interface=n_interface)

    if continue_train_r:
        print(f"loaded raw model from {raw_model_path}")
        func_params_r=checkpoint1["func_params"]
    
    time1=time.time()
    if optimizer_type=="gdlm":
        train_dic=prepare_train_r(model_r,func_params_r,data_r)
        func_params_r = gdlm_train(train_dic,LM_iter=2000,imethod=2,iaccel=1,ibold=2,continue_train=continue_train_r,model_path=raw_model_path,otherkw={"Cgoal":-1,"ftol":-1,"gtol":-1})
        rawnet = lambda x: vmap(functional_call,(None,None,0))(model_r,func_params_r,x)
    elif optimizer_type=="lm":
        train_dic=prepare_train_r(model_r,func_params_r,data_r)
        func_params_r = lm_train(train_dic,LM_iter=6000,continue_train=continue_train_r,model_path=raw_model_path)
        rawnet = lambda x: vmap(functional_call,(None,None,0))(model_r,func_params_r,x)
    elif optimizer_type=="adam":
        generator=gen_train_dic(data_r, full_batch=True, device=device)
        adam_train(generator, model_r, epoch= 1000, learning_rate=0.001, model_path=raw_model_path)
        rawnet = model_r
    elif optimizer_type=="adam_lbfgs":
        generator=gen_train_dic(data_r, full_batch=True, device=device)
        adam_train(generator, model_r, epoch= 20, learning_rate=0.001, model_path=raw_model_path)
        lbfgs_train(generator, model_r, max_iter=20000, continue_training=True, model_path=raw_model_path)
        rawnet = model_r
    time2=time.time()
    total_time=time2-time1

    eval_3d(rawnet,patchnet,g_base,reference_u,device=device)


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
