import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_samples2d(x_dict, xlim=(-1.2,1.2), ylim=None, title=None, save_name=None):
    fig=plt.figure(figsize=(8,8))
    if ylim==None:
        ylim=xlim
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    for name, points in x_dict.items():
        points=np.array(points.detach().cpu())
        plt.scatter(points[:,0],points[:,1],s=15,label=name)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_name:
        plt.savefig("data_cache/"+save_name+".jpg")
    plt.show()

def plot_samples3d(x_dict, xlim=(-1.2,1.2), ylim=None, zlim=None, title=None, save_name=None):
    fig=plt.figure(figsize=(8,8))
    if ylim==None:
        ylim=xlim
    if zlim==None:
        zlim=xlim
    ax3d = fig.add_subplot(projection='3d')
    ax3d.axis("scaled")
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    for name, points in x_dict.items():
        points=np.array(points.detach().cpu())
        ax3d.scatter(points[:,0],points[:,1],points[:,2], s=5,label=name)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_name:
        plt.savefig("data_cache/"+save_name+".jpg")
    plt.show()

def plot_results3d(x, f, title=None, save_name=None):
    fig=plt.figure(figsize=(8,8))
    ax3d1 = fig.add_subplot(projection='3d')
    if type(x)!=type(np.array(0)):
        x=np.array(x.detach().cpu())
    if type(f)!=type(np.array(0)):
        f=np.array(f.detach().cpu())
    ax3d1.scatter(x[:,0],x[:,1],f,s=5)
    if title:
        plt.title(title)
    if save_name:
        plt.savefig("data_cache/"+save_name+".jpg")
    plt.show()

def eval_2d(rawnet,patchnet,g_base,reference_u,device="cpu", save_name=None):
    x=g_base.whole_sampler(10000).astype(np.float64)
    patch_index=g_base.inside_u1(x)
    y=rawnet(torch.tensor(x).to(device)).detach().cpu().numpy()
    y[patch_index]+= patchnet(torch.tensor(x[patch_index]).to(device)).detach().cpu().numpy()
    y_real = reference_u.u_whole(torch.tensor(x)).detach().cpu().numpy()
    abs_error= np.abs(y-y_real)
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}")
    if save_name is None:
        save_name=""        
    plot_results3d(x,y_real,"ground_truth",save_name=save_name+"_ground_truth")
    plot_results3d(x,abs_error,f"abs_error, max={L_inf}",save_name=save_name+"_abs_error")
    plot_results3d(x,y,"result",save_name=save_name+"_result")

def eval_2d_composed(net,g_base,reference_u,device="cpu"):
    x=g_base.whole_sampler(10000).astype(np.float64)
    patch_index=g_base.inside_u1(x)
    
    result=net(torch.tensor(x).to(device)).detach().cpu().numpy()
    sign=np.zeros((len(result),1))
    sign[patch_index]=1.0
    y_pred=result[:,0:1]+sign*result[:,1:2]
    y_real = reference_u.u_whole(torch.tensor(x)).detach().cpu().numpy()
    abs_error= np.abs(y_pred-y_real)
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}")
    plot_results3d(x,y_real,"ground_truth")
    plot_results3d(x,abs_error,f"abs_error, max={L_inf}")
    plot_results3d(x,y_pred,"result")

def eval_3d(rawnet,patchnet,g_base,reference_u,device="cpu"):
    x=g_base.whole_sampler(100000).astype(np.float64)
    patch_index=g_base.inside_u1(x)
    y=rawnet(torch.tensor(x).to(device)).detach().cpu().numpy()
    y[patch_index]+= patchnet(torch.tensor(x[patch_index]).to(device)).detach().cpu().numpy()
    y_real = reference_u.u_whole(torch.tensor(x)).detach().cpu().numpy()
    abs_error= np.abs(y-y_real)
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    relative_l2= L_2/(np.linalg.norm(y_real)/np.sqrt(len(y_real)))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}", f"relative_L2:{relative_l2:.4e}")
    

def eval_3d_composed(net, g_base, reference_u, device="cpu"):
    x=g_base.whole_sampler(100000).astype(np.float64)
    patch_index=g_base.inside_u1(x)
    
    result=net(torch.tensor(x).to(device)).detach().cpu().numpy()
    sign=np.zeros((len(result),1))
    sign[patch_index]=1.0
    y_pred=result[:,0:1]+sign*result[:,1:2]
    y_real = reference_u.u_whole(torch.tensor(x)).detach().cpu().numpy()
    abs_error= np.abs(y_pred-y_real)
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    relative_l2= L_2/(np.linalg.norm(y_real)/np.sqrt(len(y_real)))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}, relative_L2:{relative_l2:.4e}")


def eval_2d_FD(U_pred,patchnet,g_base,reference_u,resolution,device="cpu"):
    X = Y = np.linspace(-1,1,resolution+1)
    XX, YY = np.meshgrid(X,Y,indexing='ij')
    grid_point= np.stack([XX,YY],axis=-1)
    flat_x=grid_point.reshape([-1,2])
    real_u= reference_u.u_whole(torch.tensor(flat_x)).numpy().reshape([resolution+1,resolution+1])
    patch_index=g_base.inside_u1(flat_x)
    V_pred=np.zeros((resolution+1)*(resolution+1))
    # breakpoint()
    V_pred[patch_index]=patchnet(torch.tensor(flat_x[patch_index],device=device)).flatten().detach().cpu().numpy()
    V_pred=V_pred.reshape([resolution+1,resolution+1])
    u_pred= U_pred+V_pred
    abs_error=np.abs(real_u-u_pred).flatten()
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}")
    fig=plt.figure(figsize=(8,6))
    c=plt.pcolormesh(X, Y, real_u-u_pred, cmap='RdBu')
    fig.colorbar(c)
    plt.show()

def eval_3d_FD(U_pred,patchnet,g_base,reference_u,resolution, device="cpu"):
    X = Y = Z = np.linspace(-1,1,resolution+1)
    XX, YY, ZZ = np.meshgrid(X,Y,Z,indexing='ij')
    grid_point= np.stack([XX,YY,ZZ],axis=-1)
    flat_x=grid_point.reshape([-1,3])
    real_u= reference_u.u_whole(torch.tensor(flat_x)).numpy().reshape([resolution+1,resolution+1,resolution+1])
    patch_index=g_base.inside_u1(flat_x)
    V_pred=np.zeros((resolution+1)**3)
    # breakpoint()
    V_pred[patch_index]=patchnet(torch.tensor(flat_x[patch_index]).to(device)).flatten().detach().cpu().numpy()
    V_pred=V_pred.reshape([resolution+1,resolution+1,resolution+1])
    u_pred= U_pred+V_pred
    abs_error=np.abs(real_u-u_pred).flatten()
    L_inf=np.max(abs_error)
    L_2=np.linalg.norm(abs_error)/np.sqrt(len(abs_error))
    print(f"L_inf:{L_inf:.4e}, L_2:{L_2:.4e}")
    

def view_3d_slice(z,f,geo=None,device="cpu"):
    z=np.array(z)
    assert z.ndim==1
    x=np.linspace(-1,1,100)
    y=np.linspace(-1,1,100)
    X, Y = np.meshgrid(x, y)
    gridxy=np.stack([X,Y],axis=-1).reshape([-1,2])
    for zi in z:
        Z=np.ones((len(gridxy),1))*zi
        P=np.concatenate([gridxy,Z],axis=-1)
        if geo:
            index=geo.inside(P)
            P=P[index]
            vaild_grid=gridxy[index]
        else:
            vaild_grid=gridxy
        if len(vaild_grid)>0:
            u = f(torch.tensor(P).to(device)).detach().cpu().numpy()
            plot_results3d(vaild_grid,u, title=f"slice at z = {zi}")
    
