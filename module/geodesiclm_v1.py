import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import torch
import numpy as np
import scipy
import numba   # using numba.jit is an optional choice
import copy
from typing import Any, Tuple, List, Dict, Union, Optional
import math

torch.set_default_dtype(torch.float64)

#The machine precision, the smallest magnitude and the largest magnitude
dpmpar=(2.22044604926e-16, 2.22507385852e-308,1.79769313485e+308)

converged_info={}
converged_info[1] = 'artol reached'
converged_info[2] = 'Cgoal reached'
converged_info[3] = 'gtol reached'
converged_info[4] = 'xtol reached'
converged_info[5] = 'xrtol reached'
converged_info[6] = 'ftol reached'
converged_info[7] = 'frtol reached'
converged_info[-1] = 'maxiters exeeded'
converged_info[-2] = 'maxfev exceeded'
converged_info[-3] = 'maxjev exceeded'
converged_info[-4] = 'maxaev exceeded'
converged_info[-5]=  "maxlam exceeded"
converged_info[-10] = 'User Termination '
converged_info[-11] = 'NaN Produced'


@torch.jit.script
def cholesky_solve_defactor(L,rhs):
    # solve the system L*L^T*x=rhs after chlesky decomposition, where L is a lower triangular matrix.
    return torch.linalg.solve_triangular(L.t(),torch.linalg.solve_triangular(L,rhs,upper=False),upper=True)
    

### version 1:It has been directly tranlated from the original Fortran script with slightly modifying.
#note1: func,jacobian,and Avv should return a detached tensor to save memory
#note2: imethod=1 and imethod=11 have problems for unknown reasons (TODO: fix these issues)
# Recommend hyperparameters:
#    1: imethod=0, iaccel=0, ibold=0        (standard LM)
#    2: imethod=2, iaccel=1, ibold=0  avmax = 0.75 (or 0.1 for hard problem)    (geodesiclm acceleration is on)
#    3: imethod=2, iaccel=1, ibold=2  avmax = 0.75 (or 0.1 for hard problem)    (geodesiclm acceleration + bold trial)
#    4: imethod=10, iaccel=1, ibold=2  avmax = 0.75   (trust region method for lamda and delta, with geodesiclm acceleration and bold trial)

def geodesiclm(func, x, jacobian=None, Avv=None, callback=None, \
      center_diff=False, h1=1.49012e-08, h2=0.1, damp_mode=1, \
      maxiter=200, maxfev=0, maxjev=0, maxaev=0, maxlam=-1, minlam=-1, \
      artol=0.001, Cgoal=1.49012e-11, gtol=1.49012e-12, xtol=1.49012e-11, xrtol=-1.0, ftol=1.49012e-12, frtol=-1.0, \
      print_level=2,\
      imethod=0, iaccel=1, ibold=2, ibroyden=0, \
      factoraccept=3, factorreject=2, avmax=0.75, state=None):
    """
!    The purpose of geolevmar is to minimize the sum of the squares
!    of m nonlinear functions of n variables by a modification of
!    the Levenberg-Marquardt algorithm that utilizes the geodesic
!    acceleration step correction, bold acceptance criterion, and
!    a Broyden update of the jacobian matrix.  The method employs one
!    of several possible schemes for updating the Levenberg-Marquardt
!    parameter.  The user must provide a subroutine which calcualtes
!    the functions, and optionally the jacobian and a directional 
!    derivative of the functions.  The latter two will be estimated
!    by finite differences if not supplied.
!
!    If you use this code, please acknowledge such by referencing one
!    one of the following papers in any published work:
!    
!    Transtrum M.K., Machta B.B., and Sethna J.P, Why are nonlinear
!    fits to data so challenging?  Phys. Rev. Lett. 104, 060201 (2010)
!
!    Transtrum M.K., Machta B.B., and Sethna J.P., The geometry of
!    nonlinear least squares with applications to sloppy model and
!    optimization.  Phys. Rev. E. 80, 036701 (2011)
!
!
!    func :is a user supplied subroutine which calculates the functions 
!    jacobian is a user supplied subroutine which calculates the jacobian of
!    of the functions if analytic_jac is True
!    Avv  :is a user supplied subroutine which calculates the directional
!    second derivative of the functions, analytic_Avv is True
!    fvec :is an output array of length m containing the funtion evaluation at
!    the final solution
!    fjac :is an output array of dimension(m,n) containing the jacobian evaluation
!    the final solution.  The array MATMUL( TRANSPOSE(fjac), fjac) is an estimate
!    of the covariance matrix of parameters at the final solution
!    n    :an input integer set to the number of parameters
!    m    :an input integer set to the number of functions
!    callback :a user supplied subroutine to be called after each iteration of the
!    algorithm. 
!    analytic_jac: an input boolean set to .TRUE. if the subroutine jacobian calculates
!    the jacobian.  If .FALSE. then a finite difference estimate will be used.
!
!    analytic_Avv :an input boolean set to .TRUE. if the subroutine Avv calculates
!    the directional second derivative.  If .FALSE. then a finite difference estimate
!    will be used.
!
!    center_diff :an input boolean.  If finite differences are used to estimate the jacobian
!    then center differences will used if center_diff is .TRUE., otherwise, forward
!    differences will be used.  Note that center differences are more accurate by require
!    more function evaluations.
!
!    h1: an input double precision specifying the step size for the finite difference estimates
!    of the jacobian.
!
!    h2: an input double precision specifying the steps ize for the finite difference estiamtes
!    of the directional second derivative.
!
!    dtd: a double precision array of dimension(n,n).  dtd is used as the damping matrix in the 
!    Levenberg-Marquardt routine.  It's exact treatment is specified by the damp_mode input.
!
!    damp_mode: an input integer specifying the details of the LM damping as follows:
!      damp_mode = 0: dtd is set to the identity.
!      damp_mode = 1: dtd should be a positive definite, diagonal matrix whose entries are dynamically
!                updated based on the elements of the jacobian.
!
!    niters: an output integer specifying the number of iterations of the algorithm.
!
!    nfev: an output integer specifying the number of calls to func.  
!
!    njev: an output integer specifying the number of calls to jacobian.
!
!    naev: an output integer specifying the number of calls to Avv.
!
!    maxiter: an input integer specifying the maximum number of routine iterations.
!
!    maxfev: an input integer specifying the maximum number of function calls
!    if maxfev = 0, then there is no limit to the number of function calls.
!
!    maxjev an input integer specifying the maximum number of jacobian calls
!    if maxjev = 0, then there is no limit to the number of jacobian calls.
!
!    maxaev an input integer specifying the maximum number of Avv calls
!    if maxaev = 0, then there is no limit to the number of Avv calls.
!
!    maxlam an input double precision specifying the maximum allowed value of 
!    the damping term lambda. If this is negative, then there is no limit.
!
!    minlam an input double precision specifying the minimum allowed value of 
!    the damping term lambda. If lambda is smaller than this value for three consecutive steps
!    the routine terminates.  If this is negative, then there is no limit.
!
!    artol: an input double precision.  The method will terminate when the cosine of the
!    angle between the residual vector and the range of the jacobian is less than artol.
!
!    Cgoal: an input double precision.  The method will terminate when the cost (one half
!    the sum of squares of the function) falls below Cgoal.
!
!    gtol: an input double precision.  The method will terminate when norm of Cost gradient 
!    falls below gtol.
!    
!    xtol: an input double precision.  The method will terminate when parameters change by
!    less than xtol.
!
!    xrtol: an input double precision.  The method will terminate if the relative change in
!    each of the parameters is less than xrtol.
!
!    ftol: an input double precision.  The method will termiante if the Cost fails to decrease
!    by more than ftol for 3 consecutive iterations.
!
!    frtol: an input double precision.  The method will terminate if the relative decrease in
!    Cost is less than frtol 3 consecutive iterations.
!
!    converged: an output integer indicated the reason for termination:
!      converged = 1: artol 
!      converged = 2: Cgoal
!      converged = 3: gtol
!      converged = 4: xtol
!      converged = 5: xrtol
!      converged = 6: ftol
!      converged = 7: frtol
!      converged = -1: maxiters exeeded
!      converged = -2: maxfev exceeded
!      converged = -3: maxjev exceeded
!      converged = -4: maxaev exceeded
!      converged = -10: user requested termination in callback via info
!      converged = -11: Either the initial function evalaution or subsequent jacobian
!                       evaluations produced Nans.
!    print_level: an input integer specifying the amount of details to be printed.
!    acceptable values range from 0 to 5, with larger number printing more details.
!
!    print_unit: an input integer specifying the unit number details should be written to.
!
!    iaccel = 1 : 1 or 0 for geodesic acceleration on or off.
!    
!    imethod: an input integer specifying the method for updating the LM parameter
!      imethod = 0: adjusted by fixed factors after accepted/rejected steps
!      imethod = 1: adjusted as described in Nielson
!      imethod = 2: adjusted according to an unpublished method due to Cyrus Umrigar and Peter Nightingal
!      imethod = 10: step size Delta adjusted by fixed factors after accepted/rejected steps
!      imethod = 11: step size adjusted as described in More'
!    ibold = 0 : (0-4) If nonzero this allows some uphill steps which can speed convergence. 1->4 from stricter to looser uphill acceptance.
!    
!    ibroyden = 0 : If nonzero this employs Broyden approximate Jacobian updates. 
!    Can speed up algorithm with cost of accuracy.
!
!    initialfactor: an input double precision for specifying either the initial LM parameter
!    of the initial step size.
!
!    factoraccept: an input double precision (larger than 1.0) specifying the factor by which
!    either the LM parameter or the step size will be adjusted after an accepted step if
!    imethod = 0 or 10
!
!    factorreject: an input double precision (larger than 1.0) specifying the factor by which
!    either the LM parameter of the step size will be adjusted after a rejected step if
!    imethod = 0 or 10
!
!    avmax: an input double precision specifying the maximum norm of the geodesic acceleration 
!    relative to the velocity vector. 0.75 for most situations, and 0.1 for difficult problem
!
    """
    if callback is None:
        callback= lambda *args: 0
    #  Initialize variables
    nfev= naev = njev = counter = 0
    cos_alpha = 1.0e+0
    av = 0.0e+0
    a_param = 0.5
    accepted = converged = 0
    delta=-1
    actred = 0.0; rho = 0.0
    fvec = func(x); nfev += 1
    n = x.shape[0]; m = fvec.shape[0]
    a = torch.zeros(n)  # for computing accelerator
    v=vold=torch.zeros_like(x)
    acc=torch.zeros_like(x)
    C = 0.5*torch.dot(fvec,fvec)
    if print_level >= 1:
        print(f"initial cost: {C}")
    # Check for nans in fvec
    if torch.any(fvec.isnan()):
        converged = -11; maxiter = 0
    Cold = Cbest = C

    if jacobian:
        fjac=jacobian(x)
        njev = njev + 1
    else:
        fjac=fdjac(x,fvec,func,h1,center_diff)
        if center_diff:
            nfev = nfev + 2*n
        else:
            nfev = nfev + n
    jac_uptodate = True; jac_force_update = False
    jtj = fjac.T@fjac
    #!! Check fjac for nans
    if torch.any(fjac.isnan()):
        converged = -11
        maxiter = 0
    #!! Initialize scaling matrix
    if damp_mode==0:
        dtd=torch.eye(n)
    elif damp_mode==1:
        dtd=torch.diag(torch.maximum(torch.diag(jtj),torch.ones(n).to(jtj.device)))
        diag_indices = torch.arange(dtd.shape[0], dtype=torch.long, device=dtd.device)
    #Initialize lambda
    if imethod < 10:
        initialfactor=0.001
        lam = torch.diag(jtj).max()
        lam = torch.min(lam * initialfactor,torch.tensor(1.e5))
        # lam = torch.tensor(lam)
    # Initialize step bound if using trust region method
    elif imethod >= 10:
        initialfactor=100
        delta = initialfactor*torch.sqrt(torch.dot(x,dtd@x))
        lam = 1.0
        if delta == 0.0:
            delta = 100.
        if converged == 0:
            lam=TrustRegion(fvec,fjac,dtd,delta,lam) # Do not call this if there were nans in either fvec or fjac
    # Load memory from `state` 
    # state={"xbest":xbest,"Cbest":Cbest,"lam":lam,"delta":delta,"dtd":dtd,"a_param":a_param}
    start=0
    history=[]
    if type(state)==dict:
        x=state.get("xbest",x)
        C=state.get("Cbest",C)
        lam=state.get("lam",lam)
        delta=state.get("delta",delta)
        dtd=state.get("dtd",dtd)
        a_param=state.get("a_param",a_param)
        fvec = func(x)
        start = state.get("istep",0)
        history= state.get( "history",[])  # to collect the history of cost

    # Main Loop
    fvec_new=fvec; xbest=x_new=x; Cnew=C
    try: 
        for istep in range(start,start+maxiter):
            # Update Functions
            #Full or partial Jacobian Update?   ibroyden >0:  the max attempts failed to accept
            if accepted > 0 and ibroyden == 0:
                jac_force_update = True
            if accepted + ibroyden <= 0 and not jac_uptodate:  #Force jac update after too many failed attempts
                jac_force_update = True  
            if accepted > 0 and ibroyden > 0 and not jac_force_update: #!! Rank deficient update of jacobian matrix
                fjac = updatejac(fjac, fvec, fvec_new, acc, v, a)
                jac_uptodate = False
            
            if accepted > 0: # Accepted step
                fvec = fvec_new
                x = x_new
                vold = v
                C = Cnew
                if C <= Cbest:
                    Cbest = C
                    xbest=x
            if jac_force_update: # Full rank update of jacobian
                if jacobian:
                    fjac=jacobian(x)
                    njev = njev + 1
                else:
                    fjac=fdjac(x,fvec,func,h1,center_diff)
                    if center_diff:
                        nfev = nfev + 2*n
                    else:
                        nfev = nfev + n
                jac_uptodate = True
                jac_force_update = False
            if not torch.any(fjac.isnan()): # valid_result
                jtj = fjac.T@fjac
                #  Update Scaling/lam/TrustRegion
                if istep>0:  #Only necessary after first step
                    if damp_mode==1:
                        dtd=torch.diag_embed(torch.maximum(torch.diagonal(dtd),torch.diagonal(jtj)))
                    if imethod==0:
                        lam = Updatelam_factor(lam, accepted, factoraccept, factorreject)
                    elif imethod==1:
                        lam = Updatelam_nelson(lam, accepted, factoraccept, factorreject, rho)
                    elif imethod==2:
                        lam, a_param= Updatelam_Umrigar(lam,accepted, v, vold, dtd, a_param, Cold, Cnew)
                    elif imethod==10:
                        delta=UpdateDelta_factor(delta, accepted, factoraccept, factorreject)
                        lam = TrustRegion(fvec, fjac, dtd, delta, lam)
                    elif imethod==11:
                        delta, lam = UpdateDelta_more(delta, lam, v, dtd, rho, C, Cnew, dirder, actred, av, avmax)
                        lam = TrustRegion(fvec, fjac, dtd, delta, lam)
                    if maxlam >=0: lam=min(maxlam,lam)
                    lam=max(minlam,lam)
                # Propose Step
                # metric aray
                g = jtj + lam*dtd
                
                #Cholesky decomposition    # a faster way to check if a matrix is positive-definite. 
                L, info= torch.linalg.cholesky_ex(g)  #returning the LAPACK error codes, which is faster than torch.linalg.cholesky                
                
            else:
                converged=-11
                break
            if info==0.0:
                v=-1.0*fvec@fjac  #! velocity
                v=cholesky_solve_defactor(L,v.unsqueeze(1)).squeeze()

                # Calcualte the predicted reduction and the directional derivative -- useful for updating lam methods
                temp1 = 0.5*torch.dot(v,jtj@v)/C
                temp2 = 0.5*lam*torch.dot(v,dtd@v)/C
                pred_red = temp1 + 2.0*temp2           # for calculating `rho` in Updatelam_nelson()
                dirder = -1.0*(temp1 + temp2)          # a factor used in UpdateDelta_more()
                # calculate cos_alpha -- cos of angle between step direction (in data space) and residual vector
                cos_alpha = torch.dot(fvec,fjac@v).abs()
                cos_alpha = cos_alpha/torch.sqrt( torch.dot(fvec, fvec)*torch.dot(fjac@v,fjac@v))
                if imethod < 10:
                    delta = torch.sqrt(torch.dot(v, dtd@v)) # Update delta if not set directly
                # update acceleration
                if iaccel > 0:
                    if Avv: 
                        acc=Avv(x,v)
                        naev = naev + 1
                    else:
                        acc=FDAvv(x,v,fvec, fjac, func, jac_uptodate, h2)
                        if jac_uptodate:
                            nfev = nfev + 1
                        else:
                            nfev = nfev + 2 # we don't use the jacobian if it is not up to date
                    # Check accel for nans
                    valid_result =not torch.any(acc.isnan())
                    if valid_result:
                        a = -1.0*acc@fjac
                        a = cholesky_solve_defactor(L,a.unsqueeze(1)).squeeze()
                    else:
                        a[:] = 0.0 # If nans in acc, we will ignore the acceleration term
                
                # Evaluate at proposed step -- only necessary if av <= avmax
                av = torch.sqrt(torch.dot(a,dtd@a)/torch.dot(v,dtd@v))
                if av <= avmax:
                    x_new = x + v + 0.5*a
                    fvec_new=func(x_new)
                    nfev = nfev + 1
                    Cnew = 0.5*torch.dot(fvec_new,fvec_new)
                    Cold = C
                    # Check for nans in fvec_new
                    valid_result = not torch.any(fvec_new.isnan())
                    if valid_result: # If no nans, proceed as normal
                        # update rho and actred
                        actred = 1.0 - Cnew/C      # used in UpdateDelta_more 
                        rho = 0.0                  # used in Updatelam_nelson and UpdateDelta_more 
                        if pred_red <= 0.0:
                            rho = (1.0 - Cnew/C)/pred_red
                            # Accept or Reject proposed step
                        accepted = Acceptance(C, Cnew, Cbest, ibold, accepted, dtd, v, vold)
                    else: # If nans in fvec_new, reject step
                        actred = 0.0
                        rho = 0.0
                        accepted = np.minimum(accepted - 1, -1)
                else: # If acceleration too large, then reject
                    accepted = np.minimum(accepted - 1, -1)
            else:  #If matrix factorization fails, reject the proposed step
                accepted = np.minimum(accepted - 1, -1)
            # Check Convergence
            if converged==0:
                converged, counter = convergence_check(converged, accepted, counter, \
                C, Cnew, x, fvec, fjac, lam, x_new, \
                nfev, maxfev, njev, maxjev, naev, maxaev, maxlam, minlam, \
                artol, Cgoal, gtol, xtol, xrtol, ftol, frtol,cos_alpha)
                if converged >= 1 and not jac_uptodate:  
                    # If converged by artol with an out of date jacobian, update the jacoban to confirm true convergence
                    converged = 0
                    jac_force_update = True

            if print_level == 2 and  accepted > 0 :
                # print("  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted)
                # print("  Cost, lam, delta", C, lam, delta)
                # print("  av, cos alpha", av, cos_alpha)
                print("istep, Cost, lam, delta", istep, C, lam, delta)
            elif print_level == 3:
                # print("  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted)
                # print("  Cost, lam, delta", C, lam, delta)
                # print("  av, cos alpha", av, cos_alpha)
                print("istep, Cost, lam, delta", istep, C, lam, delta)
            
            if print_level == 4 and accepted > 0:
                print("  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted)
                print("  Cost, lam, delta", C, lam, delta)
                print("  av, cos alpha", av, cos_alpha)
                print("  x = ", x)
                print("  v = ", v)
                print("  a = ", a)
                
            elif print_level == 5:
                print("  istep, nfev, njev, naev, accepted", istep, nfev, njev, naev, accepted)
                print("  Cost, lam, delta", C, lam, delta)
                print("  av, cos alpha", av, cos_alpha)
                print("  x = ", x)
                print("  v = ", v)
                print("  a = ", a)

            # If converged -- return
            if converged != 0 :
                break
            if accepted >= 0:
                jac_uptodate = False # jacobian is now out of date
            history.append(C)

    except KeyboardInterrupt:
        converged = -10

    if converged == 0: converged = -1
    niters = istep
    if print_level >= 1:
        fvec = func(xbest)
        print("Optimization finished")
        print("Results:")
        print("  Converged:    "+ converged_info[converged], converged)
        print("  Final Cost: ", 0.5*torch.dot(fvec,fvec))
        # print("  Cost/DOF: ", 0.5*torch.dot(fvec,fvec)/(m-n))
        print("  niters:     ", istep)
        print("  nfev:       ", nfev)
        print("  njev:       ", njev)
        print("  naev:       ", naev)

    state={"xbest":xbest,"Cbest":Cbest,"lam":lam,"delta":delta,"dtd":dtd,"a_param":a_param, "istep":istep, "history": history}
    return xbest, state


def fdjac(x,fvec,func,eps:float,center_diff:bool):
    n=x.shape[0]
    m=fvec.shape[0]
    dx=torch.zeros(n).to(x)
    fjac=torch.zeros(m,n).to(x)
    epsmach = 2.22044604926e-16
    if center_diff:
        for i in range(n):
            h = eps*float(abs(x[i]))
            if h < epsmach:
                h = eps
            dx[i] = 0.5*h
            temp1=func(x+dx)
            temp2=func(x-dx)
            fjac[:,i] = (temp1 - temp2)/h
            dx[i] = 0.0
    else:
        for i in range(n):
            h = eps*float(abs(x[i]))
            if (h < epsmach):
                h = eps
            dx[i] = h
            temp1= func(x+dx)
            fjac[:,i] = (temp1 - fvec)/h
            dx[i] = 0.0
    return fjac


def FDAvv(x,v,fvec, fjac, func, jac_uptodate:bool, h2:float):
    if jac_uptodate:
        xtmp = x + h2*v
        ftmp=func(xtmp)
        acc = (2.0/h2)*( (ftmp - fvec)/h2 -fjac@v)
    else: # !if jacobian not up to date, do not use jacobian in F.D. (needs one more function call)
        xtmp = x + h2*v
        ftmp = func(xtmp)
        xtmp = x - h2*v
        acc = func(xtmp)
        acc = (ftmp - 2*fvec + acc)/(h2*h2)
    return acc

@torch.jit.script
def updatejac(fjac, fvec, fvec_new, acc, v, a):
    # for rank-deficient jacobian update
    r1 = fvec + 0.5*fjac@v + 0.125*acc
    djac = 2.0*(r1 - fvec - 0.5*(fjac@v))/torch.dot(v,v)
    fjac += 0.5*djac[:,None]*v[None,:]  
    v2 = 0.5*(v + a)
    djac = 0.5*(fvec_new - r1 -fjac@v2)/torch.dot(v2,v2)
    fjac += djac[:,None]*v2[None,:]
    return fjac

def TrustRegion(fvec,fjac,dtd,delta,lam):
    #!! Parameters for dgqt
    rtol = 1.0e-03
    atol = 1.0e-03 
    itmax = 10
    jtilde = fjac/torch.sqrt(torch.diag(dtd))[None,:]
    gradCtilde = (fvec@jtilde).cpu().numpy()
    g = (jtilde.T@ jtilde).cpu().numpy()
    delta=float(delta)
    lam=float(lam)
    lam, x =dgqt(g, gradCtilde, delta, rtol, atol, itmax, lam)
    return lam

def Updatelam_factor(lam, accepted, factoraccept, factorreject):
    if accepted>=0:
        lam = lam / factoraccept
    else:
        lam = lam * factorreject
    return lam


def Updatelam_nelson(lam, accepted, factoraccept, factorreject, rho):
    if accepted>=0:
        lam = lam * max(1.0/factoraccept, 1.0 - (factorreject - 1.0)*(2.0*rho - 1.0)**3 )
    else:
        nu = factorreject
        for i in range(2,-1*accepted+1):   # double nu for each rejection
            nu = nu*2.0
        lam = lam * nu
    return lam

@torch.jit.script
def Updatelam_Umrigar(lam,accepted:int, v, vold, dtd, a_param:float, Cold, Cnew):
    eps=2.22044604926e-16
    amemory = math.exp(-1.0/5.0)
    cos_on = torch.dot(v,dtd@vold)
    cos_on = cos_on/torch.sqrt(eps+torch.dot(v,dtd@v)*torch.dot(vold, dtd@vold))
    if accepted>=0:
        if Cnew <= Cold:
            if cos_on > 0:
                a_param = amemory*a_param + 1.0 - amemory
            else:
                a_param =  amemory*a_param + 0.5*(1.0 - amemory)
        else:
            a_param =  amemory*a_param + 0.5*(1.0 - amemory)
        factor = min( 100.0, max(1.1, 1.0/(2.2e-16 + 1.0-abs(2.0*a_param - 1.0))**2))
        if Cnew <= Cold and cos_on >= 0:
            lam = lam/factor
        elif Cnew > Cold:
            lam = lam*math.sqrt(factor)
    else:
        a_param =  amemory*a_param
        factor = min( 100., max(1.1, 1.0/(2.2e-16 + 1.0-abs(2.0*a_param - 1.0))**2))
        if cos_on > 0:
            lam = lam * math.sqrt(factor)
        else:
            lam = lam * factor
    return lam, a_param

def UpdateDelta_factor(delta, accepted, factoraccept, factorreject):
    if accepted >= 0:
        delta = delta * factoraccept
    else:
        delta = delta / factorreject
    return delta


def UpdateDelta_more(delta, lam,v, dtd, rho, C, Cnew, dirder, actred, av, avmax):
    pnorm = torch.sqrt(torch.dot(v,dtd@v))
    if rho > 0.25:
        if lam > 0.0 and rho < 0.75:
            temp = 1.0
        else:
            temp = 2.0*pnorm/delta
        
    else:
        if actred >= 0.0:
            temp = 0.5
        else:
            temp = 0.5*dirder/(dirder + 0.5*actred)
        
        if 0.01*Cnew >= C or temp < 0.1: 
            temp = 0.1
    # We need to make sure that if acceleration is too big, we decrease the step size
    if av > avmax:
        temp = min(temp,max(avmax/av,0.1))

    delta = temp*min(delta,10.0*pnorm)
    lam = lam/temp
    return delta,lam

@torch.jit.script
def Acceptance(C, Cnew, Cbest, ibold:int, accepted:int, dtd, v, vold):
    if Cnew <= C: # Accept all downhill steps
        accepted = max(accepted + 1, 1)
    else:
        # Calculate beta
        if torch.dot(vold,vold) == 0.0 :
            beta = 1.0
        else:
            beta = torch.dot(v,dtd@vold)
            beta = beta/torch.sqrt(torch.dot(v,dtd@v) * torch.dot(vold,dtd@vold))
            beta = min(1.0,1.0-beta)
        if ibold==0:
             # Only downhill steps 
            if Cnew <= C:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted - 1, -1)

        elif ibold==1:
            if beta*Cnew <= Cbest:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted-1,-1)
        elif ibold==2:
            if beta*beta*Cnew <= Cbest:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted-1,-1)
        elif ibold==3:
            if beta*Cnew <= C:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted-1,-1)
        elif ibold==4:
            if beta*beta*Cnew <= C:
                accepted = max(accepted + 1, 1)
            else:
                accepted = min(accepted-1,-1)
    return accepted

# @torch.jit.script
def convergence_check(converged:int, accepted:int, counter:int, \
              C, Cnew, x, fvec, fjac, lam, xnew, \
              nfev:int, maxfev:float, njev:int, maxjev:float, naev:int, maxaev:float, maxlam:float, minlam:float, \
              artol:float, Cgoal:float, gtol:float, xtol:float, xrtol:float, ftol:float, frtol:float, cos_alpha):
    #!  The first few criteria should be checked every iteration, since
    #!  they depend on counts and the Jacobian but not the proposed step.
    # nfev
    if maxfev > 0:
        if nfev>=maxfev:
            converged = -2
            counter = 0
            return converged, counter
    #!  njev
    if maxjev > 0:
        if njev>=maxjev:
            converged = -3
            return converged, counter
    #!  naev
    if maxaev > 0:
        if naev>=maxaev:
            converged = -4
            return converged, counter
    #!  maxlam
    if maxlam > 0.0:
        if lam >= maxlam:
            counter = counter + 1
            if counter>=5:
                converged = -5
            return converged, counter
    #!  minlam
    if minlam > 0.0 and lam > 0.0:
        if lam <= minlam:
            counter = counter + 1
            if counter >= 3:
                converged = -6
            return converged, counter
    #! artol -- angle between residual vector and tangent plane
    if artol> 0.0:
        if cos_alpha <= artol:
            converged = 1
            return converged, counter
    #! If gradient is small
    grad = -1.0*fvec@fjac
    if torch.sqrt(torch.dot(grad,grad))<=gtol:
        converged = 3
        return converged, counter
    #! If cost is sufficiently small
    if C < Cgoal: # !! Check every iteration in order to catch a cost small on the first iteration
        converged = 2
        return converged, counter
    #!  If step is not accepted, then don't check remaining criteria
    if accepted<0:
        counter = 0
        converged = 0
        return converged, counter
    #! If step size is small
    if torch.sqrt(torch.dot(x-xnew,x-xnew))< xtol:
        converged = 4
        return converged, counter
    #! If each parameter is moving relatively small
    for i in range(len(x)):
        converged = 5
        if abs(x[i] - xnew[i]) > xrtol*abs(x[i]) or torch.isnan(x[i]):
            converged = 0 # continue if big step or nan in xnew
        if converged == 0:
            break
    if converged == 5:
        return converged, counter
    #! If cost is not decreasing -- this can happen by accident, so we require that it occur three times in a row
    if (C - Cnew)<= ftol and C-Cnew>=0.:
        counter = counter + 1
        if counter >= 3:
            converged = 6
            return converged, counter
        return converged, counter

    #! If cost is not decreasing relatively -- again can happen by accident so require three times in a row
    if (C - Cnew)<=(frtol*C) and (C-Cnew)>=0.:
        counter = counter + 1
        if counter >= 3:
            converged = 7
            return converged, counter
        return converged, counter
    # ! If none of the above: continue
    counter = 0
    converged = 0
    return converged, counter


## non-torch functions for trust-region methods, where numba accelerating is not available due to the use of scipy.linalg

def dgqt(a,b,delta,rtol,atol,itmax,par):
    """Given an n by n symmetric matrix A, an n-vector b, and a
c     positive number delta, this subroutine determines a vector
c     x which approximately minimizes the quadratic function
c
c           f(x) = (1/2)*x'*A*x + b'*x
c
c     subject to the Euclidean norm constraint
c
c           norm(x) <= delta.
c
c     This subroutine computes an approximation x and a Lagrange
c     multiplier par such that either par is zero and
c
c            norm(x) <= (1+rtol)*delta,
c
c     or par is positive and
c
c            abs(norm(x) - delta) <= rtol*delta.
c
c     par is a double precision variable.
c         On entry par is an initial estimate of the Lagrange
c            multiplier for the constraint norm(x) <= delta.
c         On exit par contains the final estimate of the multiplier.
c     x is a double precision array of dimension n.
c         On entry x need not be specified.
c         On exit x is set to the final estimate of the solution.
c    Originate from MINPACK-2 Project. October 1993.
c     Argonne National Laboratory
c     Brett M. Averick and Jorge J. More'.
c    Reference to J. J. Mor\'e and D. C. Sorensen,
c      Computing a trust region step,
c      SIAM J. Sci. Statist. Comput. 4 (1983), 553-572.

"""
    
    n=len(a)
    x=np.zeros(n)
    z=np.zeros(n)
    zero=0.0;p001=1.0e-3;p5=0.5;one=1.0
    parf = 0.
    xnorm = 0.
    rxnorm = 0.
    rednc = False
    wa1=np.diag(a)
    wa2=np.sum(np.abs(a),axis=0)  #the Gershgorin row sums
    anorm=np.max(wa2)  #l1-norm of A
    wa2 -= np.abs(wa1)
    bnorm = np.linalg.norm(b)  #l2-norm of b

    # Calculate a lower bound, pars, for the domain of the problem.
    #  Also calculate an upper bound, paru, and a lower bound, parl,
    #  for the Lagrange multiplier.
    pars = max(-anorm,np.max(-wa1))
    parl = max(-anorm,np.max(wa1+wa2))
    paru = max(-anorm,np.max(-wa1+wa2))
    parl = max(zero,bnorm/delta-parl,pars)
    paru = max(zero,bnorm/delta+paru)
    # If the input par lies outside of the interval (parl,paru),
    #  set par to the closer endpoint.
    par = max(par,parl)
    par = min(par,paru)

    paru = max(paru,(one+rtol)*parl) #Special case: parl = paru.
    # Beginning of an iteration.
    info = 0
    for iter in range(itmax):
        if par <= pars and paru > zero:  #  Safeguard par.
            par = max(p001,np.sqrt(parl/paru))*paru
        # Copy the lower triangle of A into its upper triangle and
        # compute A + par*I.
        a[np.diag_indices_from(a)] = wa1 + par
        
        U,indef = scipy.linalg.lapack.dpotrf(a)
        # Case 1: A + par*I is positive definite.
        if indef==0:
            # Compute an approximate solution x and save the
            # last value of par with A + par*I positive definite.
            parf=par
            wa2=scipy.linalg.blas.dtrsv(U,b,trans=1)
            rxnorm=np.linalg.norm(wa2)
            x= 0 - scipy.linalg.blas.dtrsv(U,wa2)
            xnorm =np.linalg.norm(x)
            # Test for convergence.
            if abs(xnorm-delta) <= rtol*delta or (par == zero and xnorm <= (one+rtol)*delta):
                info = 1
            # Compute a direction of negative curvature and use this
            # information to improve pars.
            rznorm, z =destsv(U)
            pars = max(pars,par-rznorm**2)
            rednc = False
            if xnorm < delta:

                # Compute alpha
                prod = np.dot(z,x)/delta
                temp = (delta-xnorm)*((delta+xnorm)/delta)
                alpha = temp/(abs(prod)+np.sqrt(prod**2+temp/delta))
                alpha = np.sign(prod)*alpha if prod!=0. else alpha

                # Test to decide if the negative curvature step
                # produces a larger reduction than with z = 0.

                rznorm = abs(alpha)*rznorm
                if (rznorm/delta)**2+par*(xnorm/delta)**2 <= par: 
                    rednc = True

                # Test for convergence.
                if p5*(rznorm/delta)**2 <=rtol*(one-p5*rtol)*(par+(rxnorm/delta)**2):
                        info = 1
                elif p5*(par+(rxnorm/delta)**2) <=(atol/delta)/delta and info==0: 
                        info = 2 
                elif xnorm == zero:
                        info = 1
            # Compute the Newton correction parc to par.
            if xnorm == zero:
                parc = -par
            else:
                wa2=x[:]/xnorm
                wa2=scipy.linalg.blas.dtrsv(U,wa2,trans=1)
                temp = np.linalg.norm(wa2)
                parc = (((xnorm-delta)/delta)/temp)/temp
            # Update parl or paru.

            if xnorm > delta: parl = max(parl,par)
            if xnorm < delta: paru = min(paru,par)

        else: #  Case 2: A + par*I is not positive definite.
            #Use the rank information from the Cholesky
            #decomposition to update par.
            if indef > 1:  
                #the leading minor of order `indef` is not
                #positive definite, so that the factorization could not becompleted.
                k= indef -1  
                # Restore column indef to A + par*I.
                U[:k,k]= a[:k,k]
                U[k,k]=par+wa1[k]

                #Compute parc.
                wa2[:k]= a[:k,k]
                wa2[:k]=scipy.linalg.blas.dtrsv(U[:k,:k],wa2[:k],trans=1)
                U[k,k] -= np.linalg.norm(wa2[:k])**2
                wa2[:k]=scipy.linalg.blas.dtrsv(U[:k,:k],wa2[:k])
            
            wa2[k] = -one
            temp = np.linalg.norm(wa2[:k+1])
            parc = -(U[k,k]/temp)/temp
            pars = max(pars,par,par+parc)

            # If necessary, increase paru slightly.
            # This is needed because in some exceptional situations
            # paru is the optimal value of par.
            paru = max(paru,(one+rtol)*pars)
        
        # Use pars to update parl.
        parl = max(parl,pars)

        # Test for termination.
        if info == 0:
            if iter == itmax: info = 4
            if paru <= (one+p5*rtol)*pars: info = 3
            if paru == zero: info = 2
        # If exiting, store the best approximation
        if info != 0:
            # Compute the best current estimates for x and f.
            par = parf
            f = -p5*(rxnorm**2+par*xnorm**2)
            if rednc:
               f = -p5*((rxnorm**2+par*delta**2)-rznorm**2)
               x += alpha*z
            return par, x
        # Compute an improved estimate for par.
        par = max(parl,par+parc)
    return par, x

@numba.jit(nopython=True)
def destsv(R):
    """
c     Given an n by n upper triangular matrix R, this subroutine
c     estimates the smallest singular value and the associated
c     singular vector of R.
c
c     In the algorithm a vector e is selected so that the solution
c     y to the system R'*y = e is large. The choice of sign for the
c     components of e cause maximal local growth in the components
c     of y as the forward substitution proceeds. The vector z is
c     the solution of the system R*z = y, and the estimate svmin
c     is norm(y)/norm(z) in the Euclidean norm.
c     Originate from MINPACK-2 Project. October 1993.
c      Argonne National Laboratory
c      Brett M. Averick and Jorge J. More'.
c     Reference to J. J. Mor\'e and D. C. Sorensen,
c      Computing a trust region step,
c      SIAM J. Sci. Statist. Comput. 4 (1983), 553-572.
    """
    zero=0.; p01=1.0e-2; one=1.
    n=len(R)
    z=np.zeros(n)
    e= R[0,0] # This choice of e makes the algorithm scale invariant.
    if e == zero:
         svmin = zero
         z[0] = one
         return svmin, z
    # Solve R'*y = e.
    for i in range(n):
        e=np.sign(-z[i])*e if z[i]!=zero else e
        # Scale y. The factor of 0.01 reduces the number of scalings.
        if abs(e-z[i]) > abs(R[i,i]):
            temp = min(p01,abs(R[i,i])/abs(e-z[i]))
            z = temp*z
            e = temp*e
        #  Determine the two possible choices of y(i).
        if R[i,i] == zero:
            w = one
            wm = one
        else:
            w = (e-z[i])/R[i,i]
            wm = -(e+z[i])/R[i,i]
        # Choose y(i) based on the predicted value of y(j) for j > i.
        s = abs(e-z[i])
        sm = abs(e+z[i])
        for j in range(i + 1, n):
            sm = sm + abs(z[j]+wm*R[i,j])
        if i < n-1:
            # store z[i+1:]+w*R[i,i+1:] in z[i+1:]
            z[i+1:]+=w*R[i,i+1:]
            s = s + np.sum(np.abs(z[i+1:]))
        if s < sm:
            temp = wm - w
            w = wm
            if i < n-1:
                z[i+1:]+=temp*R[i,i+1:]
        z[i] = w
    ynorm = np.linalg.norm(z)
    # Solve R*z = y.
    for j in range(n-1, -1, -1):

        #Scale z.
        if abs(z[j]) > abs(R[j,j]):
            temp = min(p01,abs(R[j,j])/abs(z[j]))
            z=temp*z
            ynorm = temp*ynorm
        if R[j,j] == zero:
            z[j] = one
        else:
            z[j] = z[j]/R[j,j]
        temp = -z[j]
        z[:j]+=temp*R[:j,j]
      # Compute svmin and normalize z.
    znorm = one/np.linalg.norm(z)
    svmin = ynorm*znorm
    z=znorm*z
    return svmin, z

