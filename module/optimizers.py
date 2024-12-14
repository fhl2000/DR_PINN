import torch
import numpy as np

from torch.func import vmap,grad,jacrev,functional_call
import os
import time
from .utils import params_pack,params_unpack,clear_cuda_cache
# device = torch.device('cpu')
# torch.set_default_dtype(torch.float64)
    
from .geodesiclm_v1 import geodesiclm

# standard LM 
def lm_train(train_dic, LM_iter=10000,mu_update=3,div_factor = 2 ,mul_factor = 3, continue_train= False, model_path=None):
    """
    train_dic: {name: (func,vm_args,func_args),...}, use to call vmap(func,vm_args)(func_args). 
                    See the usage of torch.func.vmap for more information.
    mu_update : update \mu every mu_update iterations
    div_factor : \mu <- \mu/div_factor when loss decreases
    mul_factor : \mu <- mul_factor*\mu when loss incerases
    """
    
    func_params=next(iter(train_dic.values()))[2][0]
    x0,signature=params_unpack(func_params)
    if continue_train:
        assert os.path.exists(model_path), f"model_path{model_path} doesn't exist!"
        checkpoint=torch.load(model_path)
        mu = checkpoint["mu"]
        loss_sum_old = checkpoint["loss_sum_old"]
        itera = checkpoint["itera"]
        history = checkpoint["history"]
        func_params = checkpoint["func_params"]
        x0,signature=params_unpack(func_params)
    else:
        mu = torch.tensor(10**5).to(x0)
        loss_sum_old = 10**10
        itera = 0
        history = []
    mul_factor = torch.tensor(mul_factor).to(x0)
    div_factor = torch.tensor(div_factor).to(x0)
    I = torch.eye(x0.shape[0]).to(x0)
    x = x0
    time1=time.time()
    count=0
    for step in range(LM_iter+1):
        # Put into loss functional to get L_vec
        L_vec=[]
        J_mat=[]
        func_params_=params_pack(x,signature)
        for loss_func,vm_args,func_args in train_dic.values():
            L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
            L_item/=np.sqrt(len(L_item))
            L_vec.append(L_item)
            per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_,*func_args[1:])
            cnt = 0
            for g in per_sample_grads.values(): 
                g = g.detach()
                J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                cnt = 1
            J_mat.append(J_item)

        J_mat=torch.cat(J_mat)
        L_vec=torch.cat(L_vec)
        with torch.no_grad():
            #compute loss
            loss=0
            for vec in L_vec:
                loss = loss + torch.sum(vec**2)
            history.append(loss.item())
            #update parameters
            J_product = J_mat.t()@J_mat
            rhs = -J_mat.t()@L_vec
            # dp = torch.linalg.solve(J_product + mu*I, rhs)
            L, info= torch.linalg.cholesky_ex(J_product + mu*I)  #returning the LAPACK error codes, which is faster than torch.linalg.cholesky
            if info==0:
                dp = torch.linalg.solve_triangular(L.t(),torch.linalg.solve_triangular(L,rhs,upper=False),upper=True).squeeze()
            else:  # reject
                mu=mu*mul_factor
                itera+=1
                continue
            x_new=x+dp

            # reject if loss increases
            loss_new=0
            func_params_=params_pack(x_new,signature)
            for loss_func,vm_args,func_args in train_dic.values():
                L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
                L_item/=np.sqrt(len(L_item))
                loss_new = loss_new + torch.sum(L_item**2)
            if loss_new > loss: # reject 
                mu=mu*mul_factor
                itera+=1
                continue
            else:
                x=x_new
                
            

        itera += 1
        if step % mu_update == 0:
            #if loss_sum_check < loss_sum_old:
            if loss < loss_sum_old:
                mu = max(mu/div_factor, 10**(-15))
                count=0
            else:
                mu = min(mul_factor*mu, 10**(3))
                if mu == 10**(3):
                    count+=1
                if count>=5:
                    print("lm algorithm bloomed up,Iter %d, Loss_Res: %.5e, mu: %.5e" % (itera, loss.item(), mu) )
                    break
            loss_sum_old = loss
                
        if step%100 == 0:
            print(
                    'Iter %d, Loss_Res: %.5e, mu: %.5e' % (itera, loss.item(), mu)
                )  
        if step% 1000 == 0 and model_path:
            func_params_=params_pack(x,signature)
            torch.save({"mu":mu, "loss_sum_old":loss_sum_old, "itera":itera, "func_params": func_params, "history": history}, model_path)
        if step == LM_iter or loss.item()<10**(-15):
            break
    # save results
    time2=time.time()
    func_params=params_pack(x,signature)
    if model_path:
        torch.save({"mu":mu, "loss_sum_old":loss_sum_old, "itera":itera, "func_params": func_params, "history": history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    return func_params
    

# 参数train_dic使用说明：是一个字典，每一项如果要使用compute_loss_fun1(params,*args_fun), 和vmap(compute_loss_fun1, (None,0,0))(params,*args_fun)
#  则 train_dic["compute_loss_fun1"]=(compute_loss_fun1,(None,0,0),(params,*args_fun))

def gdlm_train(train_dic, LM_iter=2000, imethod=2,iaccel=1 ,ibold=2, continue_train= False, model_path=None, otherkw={}):
    func_params=next(iter(train_dic.values()))[2][0]
    x0,signature=params_unpack(func_params)
    
    def func(x):
        func_params_=params_pack(x,signature)
        L_vec=[]
        for loss_func,vm_args,func_args in train_dic.values():
            L_item=vmap(loss_func,vm_args)(func_params_,*func_args[1:])
            if isinstance(L_item, torch.Tensor):  # return single residual
                L_item/=np.sqrt(len(L_item))
                L_vec.append(L_item)
            else:
                for item in L_item:           # return multiple residuals   
                    item/=np.sqrt(len(item))
                    L_vec.append(item)
        L_vec=torch.cat(L_vec)
        return L_vec.squeeze(-1).detach()
    def jacb(x):
        func_params_=params_pack(x,signature)
        J_mat=[]
        for loss_func,vm_args,func_args in train_dic.values():
            per_sample_grads = vmap(jacrev(loss_func), vm_args)(func_params_,*func_args[1:])
            cnt = 0
            for g in per_sample_grads.values(): 
                g = g.detach()
                J_item = g.view(len(g), -1) if cnt == 0 else torch.hstack([J_item, g.view(len(g), -1)])
                cnt = 1
            J_item/=np.sqrt(len(J_item))
            J_mat.append(J_item)

        J_mat=torch.cat(J_mat)
        return J_mat  # already detached

    if continue_train and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        state=checkpoint["state"]
    else: 
        state=None
    time1=time.time()
    x, state=geodesiclm(func,x0,jacb,state=state,maxiter=LM_iter,imethod=imethod,iaccel=iaccel,ibold=ibold,frtol=1e-5,print_level=2, avmax=0.1,**otherkw)
    func_params=params_pack(x,signature)
    time2=time.time()

    if model_path:
        torch.save({ "func_params": func_params, "state":state}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
    return func_params


def adam_train(gen_train_dic, model, epoch=1000, learning_rate=5.e-3,continue_training=False, model_path=None):
    if continue_training and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        opt=torch.optim.AdamW([{"params":model.parameters(),'initial_lr': learning_rate}])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        history = checkpoint['history']
        scheduler = torch.optim.lr_scheduler.StepLR(opt,last_epoch=start_epoch-1,step_size=50,gamma=0.7)
    else:
        start_epoch = 0
        opt=torch.optim.AdamW([{"params":model.parameters()}],lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=0.7)
        history = []

    time1=time.time()
    if callable(gen_train_dic):
        generator=gen_train_dic()
    else:
        generator=gen_train_dic
    train_dic = next(generator)
    for iter in range(start_epoch, start_epoch+epoch):
        # train_dic = next(generator)
        for i in range(50):
            train_dic = next(generator)
            loss=0
            for loss_fn, data in train_dic.values():
                loss = loss + loss_fn(model,*data)
            opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            opt.step()
            history.append(loss.item())
        print(f"epoch: {iter}, loss: {loss.item():.4e} ")
        scheduler.step()
        clear_cuda_cache()
        if iter%20 ==19:
            if model_path: torch.save({ "state_dict": model.state_dict(),"epoch":iter, "history": history}, model_path)


    time2=time.time()
    if model_path:
        torch.save({ "state_dict": model.state_dict(),"epoch":iter, "history":history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")

def lbfgs_train(gen_train_dic, model, max_iter=10000, continue_training=False, model_path=None):
    import itertools
    counter=itertools.count()
    if continue_training and os.path.exists(model_path):
        checkpoint=torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        history = checkpoint['history']
    else:
        history=[]
    optLbfgs=torch.optim.LBFGS(model.parameters(),
                           max_iter=max_iter,
                           tolerance_grad=1e-9,
                           tolerance_change=0,
                           line_search_fn="strong_wolfe"
                           )
    if callable(gen_train_dic):
        generator=gen_train_dic()
    else:
        generator=gen_train_dic
    def closure():
        optLbfgs.zero_grad()
        train_dic = next(generator)
        loss=0
        for loss_fn, data in train_dic.values():
            loss = loss + loss_fn(model,*data)
        loss.backward()
        history.append(loss.item())
        if next(counter)%100==0:
            clear_cuda_cache()
            print(f"iter{len(history)},loss: {loss.item():.4e}")
            if model_path: torch.save({ "state_dict": model.state_dict(),"history": history}, model_path)
            if loss.isnan():
                print("Loss is NaN, exiting the program")
                exit()
        return loss
    time1=time.time()
    optLbfgs.step(closure=closure)
    time2=time.time()
    if model_path:
        torch.save({ "state_dict": model.state_dict(),"history": history}, model_path)
        print(f"Trained results saved at {model_path}. Using time {time2-time1:.2f}s")
