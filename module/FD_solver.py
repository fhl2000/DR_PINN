import numpy as np
import scipy

def possion2d_solver(f,boundary,resolution):
    # initialize
    maxIter= 50000
    maxerr = 0.5*1e-10
    dx= 2.0/resolution
    U = np.zeros([resolution+1,resolution+1])
    X = Y = np.linspace(-1,1,resolution+1)
    XX, YY = np.meshgrid(X,Y,indexing='ij')
    grid_point= np.stack([XX,YY],axis=-1)
    # set boundary
    U[:,0] = boundary(grid_point[:,0])
    U[0,:] = boundary(grid_point[0,:])
    U[:,-1] = boundary(grid_point[:,-1])
    U[-1,:] = boundary(grid_point[-1,:])
    
    # righr hand sides
    F = f(grid_point.reshape([-1,2])).reshape([resolution+1,resolution+1])
    # begin iteration using the Jacobi method
    for iter in range(maxIter):
        U_old = U.copy()
        U[1:-1,1:-1] = 1.0/4*(U[0:-2,1:-1]+U[2:,1:-1]+U[1:-1,0:-2]+U[1:-1,2:]+dx**2*F[1:-1,1:-1])
        if np.max(abs(U-U_old))<=maxerr:
            break
    print("last iter max(abs(U-U_old)): ",np.max(abs(U-U_old)))
    return U

def FFT_poisson2D_fast_solver(f,boundary,resolution):
    dx= 2.0/resolution
    X = Y = np.linspace(-1,1,resolution+1)
    XX, YY = np.meshgrid(X,Y,indexing='ij')
    grid_point= np.stack([XX,YY],axis=-1)
    U = np.zeros([resolution+1,resolution+1])
    # set boundary
    U[:,0] = boundary(grid_point[:,0])
    U[0,:] = boundary(grid_point[0,:])
    U[:,-1] = boundary(grid_point[:,-1])
    U[-1,:] = boundary(grid_point[-1,:])
    # right hand sides matrix
    F = f(grid_point.reshape([-1,2])).reshape([resolution+1,resolution+1])
    # breakpoint()
    # correcting F matrix
    F[1,:] += U[0,:]/dx**2
    F[-2,:] += U[-1,:]/dx**2
    F[:,1] += U[:,0]/dx**2
    F[:,-2] += U[:,-1]/dx**2
    def fastsolve(F,nx,ny,dx,dy):
        lamdax=2*(1-np.cos(np.pi*np.arange(1,nx+1)/(nx+1)))/dx**2
        lamday=2*(1-np.cos(np.pi*np.arange(1,ny+1)/(ny+1)))/dy**2
        Bbar=np.sqrt(4/((nx+1)*(ny+1)))*scipy.fft.dstn(F,type=1)/4
        lamXmat,lamYmat=np.meshgrid(lamdax,lamday,indexing="ij")
        Ubar=Bbar/(lamXmat+lamYmat)
        pv=np.sqrt(4/((nx+1)*(ny+1)))*scipy.fft.dstn(Ubar,type=1)/4
        return pv
    U[1:-1,1:-1] = fastsolve(F[1:-1,1:-1],resolution-1,resolution-1,dx,dx)
    return U

def FFT_poisson3D_fast_solver(f,boundary,resolution):
    dx= 2.0/resolution
    X = Y = Z = np.linspace(-1,1,resolution+1)
    XX, YY , ZZ = np.meshgrid(X, Y, Z, indexing='ij')
    grid_point= np.stack([XX,YY,ZZ],axis=-1)
    U = np.zeros([resolution+1,resolution+1,resolution+1])
    # set boundary
    U[:,:,0] = boundary(grid_point[:,:,0].reshape([-1,3])).reshape([resolution+1,resolution+1])
    U[0,:,:] = boundary(grid_point[0,:,:].reshape([-1,3])).reshape([resolution+1,resolution+1])
    U[:,0,:] = boundary(grid_point[:,0,:].reshape([-1,3])).reshape([resolution+1,resolution+1])
    U[:,:,-1] = boundary(grid_point[:,:,-1].reshape([-1,3])).reshape([resolution+1,resolution+1])
    U[-1,:,:] = boundary(grid_point[-1,:,:].reshape([-1,3])).reshape([resolution+1,resolution+1])
    U[:,-1,:] = boundary(grid_point[:,-1,:].reshape([-1,3])).reshape([resolution+1,resolution+1])
    # right hand sides matrix
    F = f(grid_point.reshape([-1,3])).reshape([resolution+1,resolution+1,resolution+1])
    # correcting F matrix
    F[1,:,:] += U[0,:,:]/dx**2
    F[-2,:,:] += U[-1,:,:]/dx**2
    F[:,1,:] += U[:,0,:]/dx**2
    F[:,-2,:] += U[:,-1,:]/dx**2
    F[:,:,1] += U[:,:,0]/dx**2
    F[:,:,-2] += U[:,:,-1]/dx**2
    
    def fastsolve(F,nx,ny,nz,dx,dy,dz):
        lamdax=2*(1-np.cos(np.pi*np.arange(1,nx+1)/(nx+1)))/dx**2
        lamday=2*(1-np.cos(np.pi*np.arange(1,ny+1)/(ny+1)))/dy**2
        lamdaz=2*(1-np.cos(np.pi*np.arange(1,nz+1)/(nz+1)))/dz**2
        Bbar=np.sqrt(8/((nx+1)*(ny+1)*(nz+1)))*scipy.fft.dstn(F,type=1)/8
        lamXmat,lamYmat,lamZmat=np.meshgrid(lamdax,lamday,lamdaz,indexing="ij")
        Ubar=Bbar/(lamXmat+lamYmat+lamZmat)
        pv=np.sqrt(8/((nx+1)*(ny+1)*(nz+1)))*scipy.fft.dstn(Ubar,type=1)/8
        return pv
    U[1:-1,1:-1,1:-1] = fastsolve(F[1:-1,1:-1,1:-1],resolution-1,resolution-1,resolution-1,dx,dx,dx)
    return U
    
if __name__ == "__main__":
    # test 2d
    def referece_u(x):
        return np.exp(x[:,0])*np.sin(np.pi*x[:,1])
    def f(x):
        return -(1-np.pi**2)*np.exp(x[:,0])*np.sin(np.pi*x[:,1])
    boundary =  referece_u

    resolution = 256
    X = Y = np.linspace(-1,1,resolution+1)
    XX, YY = np.meshgrid(X,Y,indexing='ij')
    grid_point= np.stack([XX,YY],axis=-1)
    real_u= referece_u(grid_point.reshape([-1,2])).reshape([resolution+1,resolution+1])

    # # methods1:
    # solution1_u=possion2d_solver(f,boundary,resolution)
    # print(np.max(np.abs(real_u-solution1_u)))

    # methods2:
    solution2_u=FFT_poisson2D_fast_solver(f,boundary,resolution)
    print(np.max(np.abs(real_u-solution2_u)))

    # test 3D
    def referece_u(x):
        return np.exp(x[:,0])*np.sin(np.pi*x[:,1])*np.cos(np.pi*x[:,2])
    def f(x):
        return -(1-2*np.pi**2)*np.exp(x[:,0])*np.sin(np.pi*x[:,1])*np.cos(np.pi*x[:,2])
    boundary =  referece_u

    resolution = 256
    X = Y = Z = np.linspace(-1,1,resolution+1)
    XX, YY, ZZ= np.meshgrid(X,Y,Z,indexing='ij')
    grid_point= np.stack([XX,YY,ZZ],axis=-1)
    real_u= referece_u(grid_point.reshape([-1,3])).reshape([resolution+1,resolution+1,resolution+1])
    solution2_u=FFT_poisson3D_fast_solver(f,boundary,resolution)
    print(np.max(np.abs(real_u-solution2_u)))
