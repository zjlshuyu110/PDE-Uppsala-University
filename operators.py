import numpy as np
import scipy.linalg as splg
import scipy.sparse as spsp

######################################################################################
##                                                                                  ##
##  SBP operators for course "Scientific computing for PDEs" at Uppsala University. ##
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31    
##  Updated: 2022-09-19                                                          ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson.                                   ##
##                                                                                  ##
##  Central operators of orders 2, 4, and 6.                                        ##
##  Upwind operators of order 3, 5, and 7.                                          ##
##  Periodic explicit operators of order 2, 4, 6, 8, 10, and 12.                    ##
##  Periodic implicit operators.                                                    ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##  - Matplotlib 3.3.4                                                              ##
##                                                                                  ##
######################################################################################

# Central 1D second order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(m,h,order)
def sbp_cent_2nd(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.eye(m)
    H[0,0] = 0.5
    H[-1,-1] = 0.5
    H = h*H

    HI = np.linalg.inv(H)

    D1 = 0.5*np.diag(np.ones(m-1),1) - 0.5*np.diag(np.ones(m-1),-1)
    D1[0,0] = -1
    D1[0,1] = 1
    D1[-1,-2] = -1
    D1[-1,-1] = 1
    D1 = D1/h

    Q = np.matmul(H,D1) + 0.5*np.tensordot(e_l, e_l, axes=0) - 0.5*np.tensordot(e_r, e_r, axes=0)

    D2 = np.diag(np.ones(m-1),1) + np.diag(np.ones(m-1),-1) - 2*np.diag(np.ones(m),0)
    D2[0,0] = 1
    D2[0,1] = -2
    D2[0,2] = 1
    D2[-1,-3] = 1
    D2[-1,-2] = -2
    D2[-1,-1] = 1
    D2 = D2/(h*h)

    d_stenc = np.array([-3./2, 2, -1./2])/h
    d1_l = np.zeros(m)
    d1_l[:3] = d_stenc
    d1_r = np.zeros(m)
    d1_r[-3:] = -np.flip(d_stenc)

    M = -np.matmul(H,D2) - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0)

    H = spsp.csc_matrix(H)
    HI = spsp.csc_matrix(HI)
    D1 = spsp.csc_matrix(D1)
    D2 = spsp.csc_matrix(D2)
    e_l = spsp.csc_matrix(e_l)
    e_r = spsp.csc_matrix(e_r)
    d1_l = spsp.csc_matrix(d1_l)
    d1_r = spsp.csc_matrix(d1_r)
    return H,HI,D1,D2,e_l,e_r,d1_l,d1_r

# Central 1D fourth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(m,h,order)
def sbp_cent_4th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4,0:4] = np.diag(np.array([17/48, 59/48, 43/48, 49/48]))
    H[-4:,-4:] = np.diag(np.array([49/48, 43/48, 59/48, 17/48]))
    H=H*h;

    HI = np.linalg.inv(H)

    Q = -1/12*np.diag(np.ones(m-2),2) + 8/12*np.diag(np.ones(m-1),1) - 8/12*np.diag(np.ones(m-1),-1) + 1/12*np.diag(np.ones(m-2),-2)
    Q_U = np.array([[0, 0.59e2/0.96e2, -0.1e1/0.12e2, -0.1e1/0.32e2],[-0.59e2/0.96e2, 0, 0.59e2/0.96e2, 0],[0.1e1/0.12e2, -0.59e2/0.96e2, 0, 0.59e2/0.96e2],[0.1e1/0.32e2, 0, -0.59e2/0.96e2, 0]])
    Q[0:4,0:4] = Q_U;
    Q[-4:,-4:] = np.flipud(np.fliplr(-Q_U));

    D1 = HI@(Q - 0.5*np.tensordot(e_l, e_l, axes=0) + 1/2*np.tensordot(e_r, e_r, axes=0))


    M_U = np.array([[0.9e1/0.8e1, -0.59e2/0.48e2, 0.1e1/0.12e2, 0.1e1/0.48e2],[-0.59e2/0.48e2, 0.59e2/0.24e2, -0.59e2/0.48e2, 0],[0.1e1/0.12e2, -0.59e2/0.48e2, 0.55e2/0.24e2, -0.59e2/0.48e2],[0.1e1/0.48e2, 0, -0.59e2/0.48e2, 0.59e2/0.24e2]])
    M = -(-1/12*np.diag(np.ones(m-2),2) + 16/12*np.diag(np.ones(m-1),1) + 16/12*np.diag(np.ones(m-1),-1) - 1/12*np.diag(np.ones(m-2),-2) - 30/12*np.diag(np.ones(m),0));

    M[0:4,0:4] = M_U

    M[-4:,-4:] = np.flipud(np.fliplr(M_U))
    M=M/h;

    d_stenc = np.array([-0.11e2/0.6e1, 3, -0.3e1/0.2e1, 0.1e1/0.3e1])/h
    d1_l = np.zeros(m)
    d1_l[0:4] = d_stenc
    d1_r = np.zeros(m)
    d1_r[-4:] = np.flip(-d_stenc)

    D2 = HI@(-M - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0))

    H = spsp.csc_matrix(H)
    HI = spsp.csc_matrix(HI)
    D1 = spsp.csc_matrix(D1)
    D2 = spsp.csc_matrix(D2)
    e_l = spsp.csc_matrix(e_l)
    e_r = spsp.csc_matrix(e_r)
    d1_l = spsp.csc_matrix(d1_l)
    d1_r = spsp.csc_matrix(d1_r)
    return H,HI,D1,D2,e_l,e_r,d1_l,d1_r

# Central 1D sixth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(m,h,order)
def sbp_cent_6th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m),0);
    H[:6,:6] = np.diag(np.array([13649/43200,12013/8640,2711/4320,5359/4320,7877/8640, 43801/43200]))
    H[-6:,-6:] = np.fliplr(np.flipud(np.diag(np.array([13649/43200,12013/8640,2711/4320,5359/4320,7877/8640,43801/43200]))));
    H=H*h;

    HI = np.linalg.inv(H)

    x1 = 0.70127127127127;

    D1 = 1/60*np.diag(np.ones(m-3),3) - 9/60*np.diag(np.ones(m-2),2) + 45/60*np.diag(np.ones(m-1),1) - 45/60*np.diag(np.ones(m-1),-1) + 9/60*np.diag(np.ones(m-2),-2) - 1/60*np.diag(np.ones(m-3),-3)

    D1_bound_stencil = np.array([[-21600/13649, 43200/13649*x1-7624/40947, -172800/13649*x1 + 715489/81894, 259200/13649*x1-187917/13649, -172800/13649*x1+735635/81894, 43200/13649*x1-89387/40947, 0, 0, 0], \
        [-8640/12013*x1+7624/180195, 0, 86400/12013*x1-57139/12013, -172800/12013*x1+745733/72078, 129600/12013*x1-91715/12013,-34560/12013*x1+240569/120130, 0, 0, 0], \
        [17280/2711*x1-715489/162660, -43200/2711*x1+57139/5422, 0, 86400/2711*x1-176839/8133, -86400/2711*x1+242111/10844, 25920/2711*x1-182261/27110, 0, 0, 0], \
        [-25920/5359*x1+187917/53590, 86400/5359*x1-745733/64308, -86400/5359*x1+176839/16077, 0, 43200/5359*x1-165041/32154, -17280/5359*x1+710473/321540, 72/5359, 0, 0], \
        [ 34560/7877*x1-147127/47262, -129600/7877*x1+91715/7877, 172800/7877*x1-242111/15754, -86400/7877*x1+165041/23631, 0, 8640/7877*x1, -1296/7877, 144/7877, 0], \
        [-43200/43801*x1+89387/131403, 172800/43801*x1-240569/87602, -259200/43801*x1+182261/43801, 172800/43801*x1-710473/262806, -43200/43801*x1, 0, 32400/43801, -6480/43801, 720/43801]])

    D1[:6,:9] = D1_bound_stencil
    D1[-6:,-9:] = np.flipud(np.fliplr(-D1_bound_stencil))
    D1 = D1/h

    Q = np.matmul(H,D1) + 0.5*np.tensordot(e_l, e_l, axes=0) - 0.5*np.tensordot(e_r, e_r, axes=0)

    D2 = (2*np.diag(np.ones(m-3),3) - 27*np.diag(np.ones(m-2),2) + 270*np.diag(np.ones(m-1),1) + 270*np.diag(np.ones(m-1),-1) - 27*np.diag(np.ones(m-2),-2) + 2*np.diag(np.ones(m-3),-3) - 490*np.diag(np.ones(m),0))/180;

    D2_bound_stencil = np.array([[114170/40947, -438107/54596, 336409/40947, -276997/81894, 3747/13649, 21035/163788, 0, 0, 0], \
        [6173/5860, -2066/879, 3283/1758, -303/293, 2111/3516, -601/4395, 0, 0, 0], \
        [-52391/81330, 134603/32532, -21982/2711, 112915/16266, -46969/16266, 30409/54220, 0, 0, 0], \
        [68603/321540, -12423/10718, 112915/32154, -75934/16077, 53369/21436, -54899/160770, 48/5359, 0, 0], \
        [-7053/39385, 86551/94524, -46969/23631, 53369/15754, -87904/23631, 820271/472620, -1296/7877, 96/7877, 0], \
        [21035/525612, -24641/131403, 30409/87602, -54899/131403, 820271/525612, -117600/43801, 64800/43801, -6480/43801, 480/43801]]);

    D2[:6,:9] = D2_bound_stencil
    D2[-6:,-9:] = np.flipud(np.fliplr(D2_bound_stencil))

    D2 = D2/h**2

    d_stenc = np.array([-25/12, 4, -3, 4/3, -1/4])/h
    d1_l = np.zeros(m)
    d1_l[0:5] = d_stenc
    d1_r = np.zeros(m)
    d1_r[-5:] = np.flip(-d_stenc)

    M = -np.matmul(H,D2) - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0)

    H = spsp.csc_matrix(H)
    HI = spsp.csc_matrix(HI)
    D1 = spsp.csc_matrix(D1)
    D2 = spsp.csc_matrix(D2)
    e_l = spsp.csc_matrix(e_l)
    e_r = spsp.csc_matrix(e_r)
    d1_l = spsp.csc_matrix(d1_l)
    d1_r = spsp.csc_matrix(d1_r)
    return H,HI,D1,D2,e_l,e_r,d1_l,d1_r

# Upwind 1D third order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_3rd(m,h,order)
def sbp_upwind_3rd(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4,0:4] = np.diag(np.array([0.4347899357e10/0.12695947216e11, 0.12032349023e11/0.9521960412e10, 0.32831414215e11/0.38087841648e11, 0.6550489565e10/0.6347973608e10]))
    H[-4:,-4:] = np.fliplr(np.flipud(H[0:4,0:4]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = -1/3*np.diag(np.ones(m-1),-1) - 1/2*np.diag(np.ones(m),0) + 1*np.diag(np.ones(m-1),1) - 1/6*np.diag(np.ones(m-2),2);

    Qu = np.array([
        [-0.847e3/0.37560e5, 0.79604458492699e14/0.119214944358240e15, -0.1643521867663e13/0.14901868044780e14, -0.4160444549287e13/0.119214944358240e15],
        [-0.22671019561497e14/0.39738314786080e14, -0.6023e4/0.37560e5, 0.91628011326497e14/0.119214944358240e15, -0.749671686919e12/0.19869157393040e14],
        [0.63495586071e11/0.1241822337065e13, -0.16644840223051e14/0.39738314786080e14, -0.4311e4/0.12520e5, 0.104757273135509e15/0.119214944358240e15],
        [0.4998377065543e13/0.119214944358240e15, -0.5276507651527e13/0.59607472179120e14, -0.12476888349687e14/0.39738314786080e14, -0.5919e4/0.12520e5]])

    Qp[:4,:4] = Qu
    Qp[-4:,-4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r

# Upwind 1D fifth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_5th(m,h,order)
def sbp_upwind_5th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4,0:4] = np.diag(np.array([0.251e3/0.720e3,0.299e3/0.240e3,0.211e3/0.240e3,0.739e3/0.720e3]))
    H[-4:,-4:] = np.fliplr(np.flipud(H[0:4,0:4]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = 1/20*np.diag(np.ones(m-2),-2) - 1/2*np.diag(np.ones(m-1),-1) - 1/3*np.diag(np.ones(m),0) + np.diag(np.ones(m-1),1) - 1/4*np.diag(np.ones(m-2),2) + 1/30*np.diag(np.ones(m-3),3)
    
    Qu = np.array([
        [-0.1e1/0.120e3, 0.941e3/0.1440e4, -0.47e2/0.360e3, -0.7e1/0.480e3],
        [-0.869e3/0.1440e4, -0.11e2/0.120e3, 0.25e2/0.32e2, -0.43e2/0.360e3],
        [0.29e2/0.360e3, -0.17e2/0.32e2, -0.29e2/0.120e3, 0.1309e4/0.1440e4],
        [0.1e1/0.32e2, -0.11e2/0.360e3, -0.661e3/0.1440e4, -0.13e2/0.40e2]])

    Qp[:4,:4] = Qu
    Qp[-4:,-4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r

# Upwind 1D seventh order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_upwind_7th(m,h,order)
def sbp_upwind_7th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:6,0:6] = np.diag(np.array([0.19087e5/0.60480e5,0.84199e5/0.60480e5,0.18869e5/0.30240e5,0.37621e5/0.30240e5,0.55031e5/0.60480e5,0.61343e5/0.60480e5]))
    H[-6:,-6:] = np.fliplr(np.flipud(H[0:6,0:6]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = -1/105*np.diag(np.ones(m-3),-3) + 1/10*np.diag(np.ones(m-2),-2) - 3/5*np.diag(np.ones(m-1),-1) - 1/4*np.diag(np.ones(m),0) + np.diag(np.ones(m-1),1) - 3/10*np.diag(np.ones(m-2),2) + 1/15*np.diag(np.ones(m-3),3) - 1/140*np.diag(np.ones(m-4),4);
    
    Qu = np.array([
        [-0.265e3/0.300272e6, 0.1587945773e10/0.2432203200e10, -0.1926361e7/0.25737600e8, -0.84398989e8/0.810734400e9, 0.48781961e8/0.4864406400e10, 0.3429119e7/0.202683600e9],
        [-0.1570125773e10/0.2432203200e10, -0.26517e5/0.1501360e7, 0.240029831e9/0.486440640e9, 0.202934303e9/0.972881280e9, 0.118207e6/0.13512240e8, -0.231357719e9/0.4864406400e10],
        [0.1626361e7/0.25737600e8, -0.206937767e9/0.486440640e9, -0.61067e5/0.750680e6, 0.49602727e8/0.81073440e8, -0.43783933e8/0.194576256e9, 0.51815011e8/0.810734400e9],
        [0.91418989e8/0.810734400e9, -0.53314099e8/0.194576256e9, -0.33094279e8/0.81073440e8, -0.18269e5/0.107240e6, 0.440626231e9/0.486440640e9, -0.365711063e9/0.1621468800e10],
        [-0.62551961e8/0.4864406400e10, 0.799e3/0.35280e5, 0.82588241e8/0.972881280e9, -0.279245719e9/0.486440640e9, -0.346583e6/0.1501360e7, 0.2312302333e10/0.2432203200e10],
        [-0.3375119e7/0.202683600e9, 0.202087559e9/0.4864406400e10, -0.11297731e8/0.810734400e9, 0.61008503e8/0.1621468800e10, -0.1360092253e10/0.2432203200e10, -0.10677e5/0.42896e5]
        ])

    Qp[:6,:6] = Qu
    Qp[-6:,-6:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r
    
# Central 1D finite difference explicit and periodic operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
# 
# Use as follows:
# 
# import operators as ops
# H,Q = ops.periodic_expl(m,h,order,use_AD)
# 
def periodic_expl(m,h,order,use_AD=False):
    if order == 2:
        d = np.array([-0.5,0,0.5])
        l = 1
        r = 1
    elif order == 4:
        d = np.array([1./12,-2./3,0,2./3,-1./12])
        l = 2
        r = 2
    elif order == 6:
        d = np.array([-1./60,3./20,-3./4,0,3./4,-3./20,1./60])
        l = 3
        r = 3
    elif order == 8:
        d = np.array([1./280,-4./105,1./5,-4./5,0,4./5,-1./5,4./105,-1./280])
        l = 4
        r = 4
    elif order == 10:
        d = np.array([-1./1260,5./504,-5./84,5./21,-5./6,0,5./6,-5./21,5./84,-5./504,1./1260])
        l = 5
        r = 5
    elif order == 12:
        d = np.array([1./5544,-1./385,1./56,-5./63,15./56,-6./7,0,6./7,-15./56,5./63,-1./56,1./385,-1./5544])
        l = 6
        r = 6
    else:
        raise NotImplementedError('Order not implemented.')

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v),1),v))
    H = spsp.csc_matrix(h*np.eye(m))

    if use_AD:
        if order == 2:
            d = np.array([1,-2,1])
            l = 1
            r = 1
            a = 0.5
        elif order == 4:
            d = -np.array([1, -4, 6, -4, 1])
            l = 2
            r = 2
            a = 1./12
        elif order == 6:
            d = np.array([1, -6, 15, -20, 15, -6, 1])
            l = 3
            r = 3
            a = 1./60
        elif order == 8:
            d = -np.array([1, -8, 28, -56, 70, -56, 28, -8, 1])
            l = 4
            r = 4
            a = 1./280
        elif order == 10:
            d = np.array([1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1])
            l = 5
            r = 5
            a = 1./1260
        elif order == 12:
            d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66,-12, 1])
            l = 6
            r = 6
            a = 1./5544
        else:
            raise NotImplementedError('Order not implemented.')

        v = np.zeros(m)
        for i in range(r+1):
            v[i] = d[i+l]
        for i in range(l):
            v[m-i-1] = d[l-i-1]

        S = spsp.csc_matrix(a*splg.toeplitz(np.roll(np.flip(v),1),v))

        Q = Q - S            

    return H,Q

# Central 1D finite difference implicit and periodic operators. 
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
# 
# Use as follows:
# 
# import operators as ops
# H,Q = ops.periodic_imp(m,h,use_AD)
# 
def periodic_imp(m,h,use_AD=False):

    h0 = 4203267613564094932432577824954./7049220443079284250976145948443;
    h1 = 22618790744689935699264926210401./84590645316951411011713751381316;
    h2 = -2209778222820418388602425303685./42295322658475705505856875690658;
    h3 = -1581945765./75409415044;
    h4 = 228992488./33235651987;
    h5 = 27214243./33751459947;

    q1 = 9607266784889201296177./19560081711822931675052;
    q2 = 8866705546306148289391./97800408559114658375260;
    q3 = -19659090145677941034997./293401225677343975125780;
    q4 = 127051314./37983174851;
    q5 = 389910724./128741750713;

    # Q
    l = 5
    r = 5
    d = np.array([-q5, -q4, -q3, -q2, -q1, 0, q1, q2, q3, q4, q5])

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v),1),v))

    # H
    l = 5
    r = 5
    d = np.array([h5, h4, h3, h2, h1, h0, h1, h2, h3, h4, h5])

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    H = spsp.csc_matrix(h*splg.toeplitz(np.roll(np.flip(v),1),v))

    if use_AD:
        d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66,-12, 1])
        l = 6
        r = 6
        a = 1./5544

        v = np.zeros(m)
        for i in range(r+1):
            v[i] = d[i+l]
        for i in range(l):
            v[m-i-1] = d[l-i-1]

        S = spsp.csc_matrix(a*splg.toeplitz(np.roll(np.flip(v),1),v))

        Q = Q - S

    return H,Q

# Central 1D finite difference variable coefficient second derivative periodic operator.
# Constructed from the corresponding D1 operators as follows: D2(c) = D1*diag(c)*D1.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
# 
# Output:
#   M_fun - function computing the variable coefficient matrix M
# 
# Use as follows:
# 
# import operators as ops
# M_fun = ops.periodic_variable_wide(m,h,order)
# 
def periodic_variable_wide(m,h,order):
    H,Q = periodic_expl(m,h,order,False)
    def M_fun(c):
        C = np.diag(c)
        M = -1/h*Q@C@Q
        return M

    return M_fun

# Central 1D 6th order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   D1 - first derivative SBP operator
#   D2 - second derivative SBP operator
#   D3 - third derivative SBP operator
#   D4 - fourth derivative SBP operator
#   D5 - fifth derivative SBP operator
#   e_l,e_r - vectors to extract the boundary grid points
#   d1_l,d1_r - vectors to extract the first derivatives at the boundary grid points
#   d2_l,d2_r - vectors to extract the second derivatives at the boundary grid points
#   d3_l,d3_r - vectors to extract the third derivatives at the boundary grid points
#   d4_l,d4_r - vectors to extract the fourth derivatives at the boundary grid points
# 
# Use as follows:
# 
# import operators as ops
# H,HI,D1,D2,D3,D4,D5,e_l,e_r,d1_l,d1_r,d2_l,d2_r,d3_l,d3_r,d4_l,d4_r = ops.sbp_higher_6th(m,h)
def sbp_higher_6th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m),0);
    H[:8,:8] = np.diag(np.array([0.318365e6/0.1016064e7,0.145979e6/0.103680e6,0.139177e6/0.241920e6,0.964969e6/0.725760e6,0.593477e6/0.725760e6,0.52009e5/0.48384e5,0.141893e6/0.145152e6,0.1019713e7/0.1016064e7]))
    H[-8:,-8:] = np.fliplr(np.flipud(H[:8,:8]));
    H=H*h;

    HI = np.linalg.inv(H)
    
    # D1
    q1 = 3/4
    q2 = -3/20
    q3 = 1/60
    
    Q=q3*(np.diag(np.ones(m-3),3) - np.diag(np.ones(m-3),-3)) + q2*(np.diag(np.ones(m-2),2) - np.diag(np.ones(m-2),-2)) + q1*(np.diag(np.ones(m-1),1) - np.diag(np.ones(m-1),-1));

    Q_U = np.array([
        [0, 0.1547358409e10/0.2421619200e10, -0.422423e6/0.11211200e8, -0.1002751721e10/0.8717829120e10, -0.15605263e8/0.484323840e9, 0.1023865e7/0.24216192e8, 0.291943739e9/0.21794572800e11, -0.24659e5/0.2534400e7],
        [-0.1547358409e10/0.2421619200e10, 0, 0.23031829e8/0.62899200e8, 0.10784027e8/0.34594560e8, 0.2859215e7/0.31135104e8, -0.45982103e8/0.345945600e9, -0.26681e5/0.1182720e7, 0.538846039e9/0.21794572800e11],
        [0.422423e6/0.11211200e8, -0.23031829e8/0.62899200e8, 0, 0.28368209e8/0.69189120e8, -0.9693137e7/0.69189120e8, 0.1289363e7/0.17740800e8, -0.39181e5/0.5491200e7, -0.168647e6/0.24216192e8],
        [0.1002751721e10/0.8717829120e10, -0.10784027e8/0.34594560e8, -0.28368209e8/0.69189120e8, 0, 0.5833151e7/0.10644480e8, 0.4353179e7/0.69189120e8, 0.2462459e7/0.155675520e9, -0.215471e6/0.10762752e8],
        [0.15605263e8/0.484323840e9, -0.2859215e7/0.31135104e8, 0.9693137e7/0.69189120e8, -0.5833151e7/0.10644480e8, 0, 0.7521509e7/0.13837824e8, -0.1013231e7/0.11531520e8, 0.103152839e9/0.8717829120e10],
        [-0.1023865e7/0.24216192e8, 0.45982103e8/0.345945600e9, -0.1289363e7/0.17740800e8, -0.4353179e7/0.69189120e8, -0.7521509e7/0.13837824e8, 0, 0.67795697e8/0.98841600e8, -0.17263733e8/0.151351200e9],
        [-0.291943739e9/0.21794572800e11, 0.26681e5/0.1182720e7, 0.39181e5/0.5491200e7, -0.2462459e7/0.155675520e9, 0.1013231e7/0.11531520e8, -0.67795697e8/0.98841600e8, 0, 0.1769933569e10/0.2421619200e10],
        [0.24659e5/0.2534400e7, -0.538846039e9/0.21794572800e11, 0.168647e6/0.24216192e8, 0.215471e6/0.10762752e8, -0.103152839e9/0.8717829120e10, 0.17263733e8/0.151351200e9, -0.1769933569e10/0.2421619200e10, 0]
        ])

    Q[:8,:8] = Q_U
    Q[-8:,-8:] = np.flipud(np.fliplr(-Q_U))

    D1 = HI@(Q - 0.5*np.tensordot(e_l, e_l, axes=0) + 1/2*np.tensordot(e_r, e_r, axes=0))

    # D2
    m0 = 49/18
    m1 = -3/2
    m2 = 3/20
    m3 = -1/90

    M = m3*(np.diag(np.ones(m-3),3) + np.diag(np.ones(m-3),-3)) + m2*(np.diag(np.ones(m-2),2) + np.diag(np.ones(m-2),-2)) + m1*(np.diag(np.ones(m-1),1) + np.diag(np.ones(m-1),-1)) + m0*np.diag(np.ones(m),0);

    M_U = np.array([
        [0.4347276223e10/0.3736212480e10, -0.1534657609e10/0.1210809600e10, 0.68879e5/0.3057600e7, 0.1092927401e10/0.13076743680e11, 0.18145423e8/0.968647680e9, -0.1143817e7/0.60540480e8, -0.355447739e9/0.65383718400e11, 0.56081e5/0.16473600e8],
        [-0.1534657609e10/0.1210809600e10, 0.42416226217e11/0.18681062400e11, -0.228654119e9/0.345945600e9, -0.12245627e8/0.34594560e8, -0.2995295e7/0.46702656e8, 0.52836503e8/0.691891200e9, 0.119351e6/0.12812800e8, -0.634102039e9/0.65383718400e11],
        [0.68879e5/0.3057600e7, -0.228654119e9/0.345945600e9, 0.5399287e7/0.4193280e7, -0.24739409e8/0.34594560e8, 0.7878737e7/0.69189120e8, -0.1917829e7/0.31449600e8, 0.39727e5/0.3660800e7, 0.10259e5/0.4656960e7],
        [0.1092927401e10/0.13076743680e11, -0.12245627e8/0.34594560e8, -0.24739409e8/0.34594560e8, 0.7780367599e10/0.3736212480e10, -0.70085363e8/0.69189120e8, -0.500209e6/0.6289920e7, -0.311543e6/0.17962560e8, 0.278191e6/0.21525504e8],
        [0.18145423e8/0.968647680e9, -0.2995295e7/0.46702656e8, 0.7878737e7/0.69189120e8, -0.70085363e8/0.69189120e8, 0.7116321131e10/0.3736212480e10, -0.545081e6/0.532224e6, 0.811631e6/0.11531520e8, -0.84101639e8/0.13076743680e11],
        [-0.1143817e7/0.60540480e8, 0.52836503e8/0.691891200e9, -0.1917829e7/0.31449600e8, -0.500209e6/0.6289920e7, -0.545081e6/0.532224e6, 0.324760747e9/0.138378240e9, -0.65995697e8/0.49420800e8, 0.1469203e7/0.13759200e8],
        [-0.355447739e9/0.65383718400e11, 0.119351e6/0.12812800e8, 0.39727e5/0.3660800e7, -0.311543e6/0.17962560e8, 0.811631e6/0.11531520e8, -0.65995697e8/0.49420800e8, 0.48284442317e11/0.18681062400e11, -0.1762877569e10/0.1210809600e10],
        [0.56081e5/0.16473600e8, -0.634102039e9/0.65383718400e11, 0.10259e5/0.4656960e7, 0.278191e6/0.21525504e8, -0.84101639e8/0.13076743680e11, 0.1469203e7/0.13759200e8, -0.1762877569e10/0.1210809600e10, 0.10117212851e11/0.3736212480e10]
        ])

    M[:8,:8] = M_U
    M[-8:,-8:] = np.flipud(np.fliplr(M_U))
    M = M/h

    d1_U = np.array([-0.25e2/0.12e2, 4, -3, 0.4e1/0.3e1, -0.1e1/0.4e1])/h
    d1_l = np.zeros(m)
    d1_l[:5] = d1_U
    d1_r = np.zeros(m)
    d1_r[-5:] = np.flip(-d1_U)

    D2 = HI@(-M - np.tensordot(e_l, d1_l, axes=0) + np.tensordot(e_r, d1_r, axes=0))

    # D3
    q1 = -61/30
    q2 = 169/120
    q3 = -3/10
    q4 = 7/240

    Q3=q4*(np.diag(np.ones(m-4),4) - np.diag(np.ones(m-4),-4)) + q3*(np.diag(np.ones(m-3),3) - np.diag(np.ones(m-3),-3)) + q2*(np.diag(np.ones(m-2),2) - np.diag(np.ones(m-2),-2)) + q1*(np.diag(np.ones(m-1),1) - np.diag(np.ones(m-1),-1))  

    Q3_U = np.array([
        [0, -0.10882810591e11/0.5811886080e10, 0.398713069e9/0.132088320e9, -0.1746657571e10/0.1162377216e10, 0.56050639e8/0.145297152e9, -0.11473393e8/0.1162377216e10, -0.38062741e8/0.1452971520e10, 0.30473e5/0.4392960e7],
        [0.10882810591e11/0.5811886080e10, 0, -0.3720544343e10/0.830269440e9, 0.767707019e9/0.207567360e9, -0.1047978301e10/0.830269440e9, 0.1240729e7/0.14826240e8, 0.6807397e7/0.55351296e8, -0.50022767e8/0.1452971520e10],
        [-0.398713069e9/0.132088320e9, 0.3720544343e10/0.830269440e9, 0, -0.2870078009e10/0.830269440e9, 0.74962049e8/0.29652480e8, -0.12944857e8/0.30750720e8, -0.17846623e8/0.103783680e9, 0.68707591e8/0.1162377216e10],
        [0.1746657571e10/0.1162377216e10, -0.767707019e9/0.207567360e9, 0.2870078009e10/0.830269440e9, 0, -0.727867087e9/0.276756480e9, 0.327603877e9/0.207567360e9, -0.175223717e9/0.830269440e9, 0.1353613e7/0.726485760e9],
        [-0.56050639e8/0.145297152e9, 0.1047978301e10/0.830269440e9, -0.74962049e8/0.29652480e8, 0.727867087e9/0.276756480e9, 0, -0.1804641793e10/0.830269440e9, 0.311038417e9/0.207567360e9, -0.1932566239e10/0.5811886080e10],
        [0.11473393e8/0.1162377216e10, -0.1240729e7/0.14826240e8, 0.12944857e8/0.30750720e8, -0.327603877e9/0.207567360e9, 0.1804641793e10/0.830269440e9, 0, -0.1760949511e10/0.830269440e9, 0.2105883973e10/0.1452971520e10],
        [0.38062741e8/0.1452971520e10, -0.6807397e7/0.55351296e8, 0.17846623e8/0.103783680e9, 0.175223717e9/0.830269440e9, -0.311038417e9/0.207567360e9, 0.1760949511e10/0.830269440e9, 0, -0.1081094773e10/0.528353280e9],
        [-0.30473e5/0.4392960e7, 0.50022767e8/0.1452971520e10, -0.68707591e8/0.1162377216e10, -0.1353613e7/0.726485760e9, 0.1932566239e10/0.5811886080e10, -0.2105883973e10/0.1452971520e10, 0.1081094773e10/0.528353280e9, 0]
        ])

    Q3[:8,:8] = Q3_U
    Q3[-8:,-8:] = np.flipud(np.fliplr(-Q3_U))
    Q3 = Q3/h**2

    d2_U = np.array([0.35e2/0.12e2, -0.26e2/0.3e1, 0.19e2/0.2e1, -0.14e2/0.3e1, 0.11e2/0.12e2])/h**2;
    d2_l = np.zeros(m)
    d2_l[:5] = d2_U
    d2_r = np.zeros(m)
    d2_r[-5:] = np.flip(d2_U)

    D3 = HI@(Q3 - np.tensordot(e_l, d2_l, axes=0) + np.tensordot(e_r, d2_r, axes=0) + 0.5*np.tensordot(d1_l, d1_l, axes=0) - 0.5*np.tensordot(d1_r, d1_r, axes=0))

    # D4
    m0 = 91/8
    m1 = -122/15
    m2 = 169/60
    m3 = -2/5
    m4 = 7/240

    M4 = m4*(np.diag(np.ones(m-4),4) + np.diag(np.ones(m-4),-4)) + m3*(np.diag(np.ones(m-3),3) + np.diag(np.ones(m-3),-3)) + m2*(np.diag(np.ones(m-2),2) + np.diag(np.ones(m-2),-2)) + m1*(np.diag(np.ones(m-1),1) + np.diag(np.ones(m-1),-1)) + m0*np.diag(np.ones(m),0)

    M4_U = np.array([
        [0.40833734273e11/0.10761070320e11, -0.162181998421e12/0.16397821440e11, 0.4696168417e10/0.521748864e9, -0.245714671483e12/0.68870850048e11, 0.2185939219e10/0.2869618752e10, -0.15248255797e11/0.114784750080e12, 0.345156907e9/0.12298366080e11, 0.6388381e7/0.1093188096e10],
        [-0.162181998421e12/0.16397821440e11, 0.147281127041e12/0.5380535160e10, -0.3072614435609e13/0.114784750080e12, 0.320122985851e12/0.28696187520e11, -0.768046031383e12/0.344354250240e12, 0.7861605187e10/0.14348093760e11, -0.803762437e9/0.4251287040e10, 0.167394281e9/0.86088562560e11],
        [0.4696168417e10/0.521748864e9, -0.3072614435609e13/0.114784750080e12, 0.139712483333e12/0.4782697920e10, -0.1634124842747e13/0.114784750080e12, 0.90855193447e11/0.28696187520e11, -0.26412188989e11/0.38261583360e11, 0.668741173e9/0.1793511720e10, -0.132673781e9/0.2342545920e10],
        [-0.245714671483e12/0.68870850048e11, 0.320122985851e12/0.28696187520e11, -0.1634124842747e13/0.114784750080e12, 0.437353997177e12/0.43044281280e11, -0.172873969321e12/0.38261583360e11, 0.34759553483e11/0.28696187520e11, -0.98928859751e11/0.344354250240e12, 0.295000207e9/0.3587023440e10],
        [0.2185939219e10/0.2869618752e10, -0.768046031383e12/0.344354250240e12, 0.90855193447e11/0.28696187520e11, -0.172873969321e12/0.38261583360e11, 0.126711914423e12/0.21522140640e11, -0.520477408939e12/0.114784750080e12, 0.49581230003e11/0.28696187520e11, -0.99640101991e11/0.344354250240e12],
        [-0.15248255797e11/0.114784750080e12, 0.7861605187e10/0.14348093760e11, -0.26412188989e11/0.38261583360e11, 0.34759553483e11/0.28696187520e11, -0.520477408939e12/0.114784750080e12, 0.19422074929e11/0.2391348960e10, -0.772894368601e12/0.114784750080e12, 0.10579712849e11/0.4099455360e10],
        [0.345156907e9/0.12298366080e11, -0.803762437e9/0.4251287040e10, 0.668741173e9/0.1793511720e10, -0.98928859751e11/0.344354250240e12, 0.49581230003e11/0.28696187520e11, -0.772894368601e12/0.114784750080e12, 0.456715296239e12/0.43044281280e11, -0.915425403107e12/0.114784750080e12],
        [0.6388381e7/0.1093188096e10, 0.167394281e9/0.86088562560e11, -0.132673781e9/0.2342545920e10, 0.295000207e9/0.3587023440e10, -0.99640101991e11/0.344354250240e12, 0.10579712849e11/0.4099455360e10, -0.915425403107e12/0.114784750080e12, 0.488029542379e12/0.43044281280e11]
        ])

    M4[:8,:8] = M4_U
    M4[-8:,-8:] = np.flipud(np.fliplr(M4_U))
    M4 = M4/h**3

    d3_U = np.array([-0.5e1/0.2e1, 9, -12, 7, -0.3e1/0.2e1])/h**3
    d3_l = np.zeros(m)
    d3_l[:5] = d3_U
    d3_r = np.zeros(m)
    d3_r[-5:] = np.flip(-d3_U)

    D4 = HI@(M4 - np.tensordot(e_l, d3_l, axes=0) + np.tensordot(e_r, d3_r, axes=0) + np.tensordot(d1_l, d2_l, axes=0) - np.tensordot(d1_r, d2_r, axes=0))

    # D5
    q1 = 323/48
    q2 = -13/2
    q3 = 87/32
    q4 = -19/36
    q5 = 13/288 

    Q5 = q5*(np.diag(np.ones(m-5),5) - np.diag(np.ones(m-5),-5)) + q4*(np.diag(np.ones(m-4),4) - np.diag(np.ones(m-4),-4)) + q3*(np.diag(np.ones(m-3),3) - np.diag(np.ones(m-3),-3)) + q2*(np.diag(np.ones(m-2),2) - np.diag(np.ones(m-2),-2)) + q1*(np.diag(np.ones(m-1),1) - np.diag(np.ones(m-1),-1))

    Q5_U = np.array([
        [0, 0.23397181898989e14/0.14864313784320e14, -0.1017670361831e13/0.1300627456128e13, -0.174874906777601e15/0.20810039298048e14, 0.14414141131333e14/0.867084970752e12, -0.286734308799859e15/0.20810039298048e14, 0.75017115155699e14/0.13006274561280e14, -0.20830884692939e14/0.20810039298048e14],
        [-0.23397181898989e14/0.14864313784320e14, 0, -0.589345832629e12/0.78647162880e11, 0.5300252531713e13/0.106173669888e12, -0.12975155532199e14/0.141564893184e12, 0.1166422899953e13/0.14746343040e11, -0.14621825894129e14/0.424694679552e12, 0.7595761987403e13/0.1238692815360e13],
        [0.1017670361831e13/0.1300627456128e13, 0.589345832629e12/0.78647162880e11, 0, -0.253985238283411e15/0.2972862756864e13, 0.23532911459153e14/0.123869281536e12, -0.59927151303553e14/0.330318084096e12, 0.156446056818143e15/0.1858039223040e13, -0.108213798165373e15/0.6936679766016e13],
        [0.174874906777601e15/0.20810039298048e14, -0.5300252531713e13/0.106173669888e12, 0.253985238283411e15/0.2972862756864e13, 0, -0.53734174314235e14/0.330318084096e12, 0.151345198787303e15/0.743215689216e12, -0.314163364959287e15/0.2972862756864e13, 0.3373980858359e13/0.162578432016e12],
        [-0.14414141131333e14/0.867084970752e12, 0.12975155532199e14/0.141564893184e12, -0.23532911459153e14/0.123869281536e12, 0.53734174314235e14/0.330318084096e12, 0, -0.100191165375773e15/0.990954252288e12, 0.2087867236243e13/0.30967320384e11, -0.94040302786627e14/0.6936679766016e13],
        [0.286734308799859e15/0.20810039298048e14, -0.1166422899953e13/0.14746343040e11, 0.59927151303553e14/0.330318084096e12, -0.151345198787303e15/0.743215689216e12, 0.100191165375773e15/0.990954252288e12, 0, -0.237006307377001e15/0.14864313784320e14, 0.236459906167e12/0.1734169941504e13],
        [-0.75017115155699e14/0.13006274561280e14, 0.14621825894129e14/0.424694679552e12, -0.156446056818143e15/0.1858039223040e13, 0.314163364959287e15/0.2972862756864e13, -0.2087867236243e13/0.30967320384e11, 0.237006307377001e15/0.14864313784320e14, 0, 0.44838401866451e14/0.8003861268480e13],
        [0.20830884692939e14/0.20810039298048e14, -0.7595761987403e13/0.1238692815360e13, 0.108213798165373e15/0.6936679766016e13, -0.3373980858359e13/0.162578432016e12, 0.94040302786627e14/0.6936679766016e13, -0.236459906167e12/0.1734169941504e13, -0.44838401866451e14/0.8003861268480e13, 0]
        ])

    Q5[:8,:8] = Q5_U
    Q5[-8:,-8:] = np.flipud(np.fliplr(-Q5_U))
    Q5 = Q5/h**4

    d4_U = np.array([1, -4, 6, -4, 1])/h**4
    d4_l = np.zeros(m)
    d4_l[:5] = d4_U
    d4_r = np.zeros(m)
    d4_r[-5:] = np.flip(d4_U)

    D5 = HI@(Q5 - np.tensordot(e_l, d4_l, axes=0) + np.tensordot(e_r, d4_r, axes=0) + np.tensordot(d1_l, d3_l, axes=0) - np.tensordot(d1_r, d3_r, axes=0) - 0.5*np.tensordot(d2_l, d2_l, axes=0) + 0.5*np.tensordot(d2_r, d2_r, axes=0))

    return H,HI,D1,D2,D3,D4,D5,e_l,e_r,d1_l,d1_r,d2_l,d2_r,d3_l,d3_r,d4_l,d4_r