import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from module.gradient import jacobian,hessian



mse_loss = torch.nn.MSELoss(reduction="mean")
def loss_boundary_coupling(net, X_bd, U_bd, weight_bd):
    out = net(X_bd)
    U_pred = out[:,0:1]
    return mse_loss(U_pred,U_bd)*weight_bd 

def loss_f_coupling(net, X_in, U_f, Sign, Beta, Gradbeta, weight_f):
    out = net(X_in)
    
    y=out[:,0:1]+Sign*out[:,1:2]
    grad_y= jacobian(y,X_in)
    lhs =0 # left hand side
    for k in range(X_in.shape[1]):
        lhs = lhs + hessian(y,X_in,i=k,j=k,grad_y=grad_y)
    lhs= - lhs*Beta - (grad_y*Gradbeta).sum(dim=1,keepdim=True)  # (batch, 1)
    
    return mse_loss(lhs, U_f)*weight_f

def loss_jump_coupling(net, X_interf, U_w):
    return mse_loss(net(X_interf)[:,1:2],-U_w)

def loss_normal_jump_coupling(net, X_interf, Normal_ij, U_v, iBeta1, iBeta2):
    out = net(X_interf)
    U_pred=out[:,0:1]
    V_pred=out[:,1:2]
    grad_U=jacobian(U_pred,X_interf)
    grad_V=jacobian(V_pred,X_interf)
    pred_nj = (((iBeta2-iBeta1)*grad_U-iBeta1*grad_V)*Normal_ij).sum(dim=1,keepdim=True)
    return mse_loss(pred_nj,U_v)



def loss_boundary_decoupling(net, X_bd, U_bd, weight_bd):
    out = net(X_bd)
    return mse_loss(out,U_bd)*weight_bd 

def loss_f_decoupling(net, X_in, U_f, Beta, Gradbeta, weight_f):
    y = net(X_in)
    grad_y = jacobian(y,X_in)
    lhs =0 # left hand side
    for k in range(X_in.shape[1]):
        lhs = lhs + hessian(y,X_in,i=k,j=k,grad_y=grad_y)
    lhs= - lhs*Beta - (grad_y*Gradbeta).sum(dim=1,keepdim=True)  # (batch, 1)
    
    return mse_loss(lhs, U_f)*weight_f


from setting import phi1
from torch.func import functional_call, vmap

parallel_phi1 = lambda x: vmap(phi1,(0,))(x)

def loss_normal_jump_decoupling(net, X_interf, Normal_ij, U_v, iBeta1, iBeta2):
    
    z_out =  parallel_phi1(X_interf).unsqueeze(-1)
    z_in = - z_out

    pred_out= net(torch.hstack([X_interf,z_out]))
    pred_in= net(torch.hstack([X_interf,z_in]))
    grad_out= jacobian(pred_out,X_interf)
    grad_in= jacobian(pred_in,X_interf)

    pred_nj = ((iBeta2*grad_out-iBeta1*grad_in)*Normal_ij).sum(dim=1,keepdim=True)
    return mse_loss(pred_nj,U_v)