import copy
import numpy as np
# sympy order matters; it overrides scipy (???)
import sympy as sym
import scipy
import scipy.signal
import scipy.integrate
from scipy.linalg import solve_continuous_are
from matplotlib import pyplot as plt

# define constants
M = sym.Symbol("M")
m1 = sym.Symbol("m1")
m2 = sym.Symbol("m2")
l1 = sym.Symbol("l1")
l2 = sym.Symbol("l2")
g = sym.Symbol("g")

# define variables 
t = sym.Symbol("t")
x = sym.Function("x")(t)
dx = sym.Function("dx")(t)
t1 = sym.Function("theta1")(t)
dt1 = sym.Function("dtheta1")(t)
t2 = sym.Function("theta2")(t)
dt2 = sym.Function("dtheta2")(t)
vars_ = [x,dx,t1,dt1,t2,dt2]

# other variables
ddx = dx.diff(t)
ddt1 = dt1.diff(t)
ddt2 = dt2.diff(t)
F = sym.Symbol("F")
T = 0.5 * (M + m1 + m2)*dx**2 - m1*l1*dx*dt1*sym.cos(t1) + 0.5*m1*l1**2*dt1**2 - m2*l2*dx*dt2*sym.cos(t2) + 0.5*m2*l2**2*dt2**2
V = -m1*g*l1*sym.cos(t1) - m2*g*l2*sym.cos(t2)

# solve initial conditions and costs
Times = np.arange(0,120,1e-3);
IC = np.array([0,1e-3,0,5e-5,0,1e-4]);
Q = np.diag([1,100,1,10,1,10])
R = 0.001

if __name__ == "__main__":
    ###############################################################################################
    ######################################## Part A ###############################################
    ###############################################################################################
    # compute Lagrangian
    L = T - V
    # compose equations of motion from Lagrangian
    EOM = [
            sym.diff(sym.diff(L,dx),t) - sym.diff(L,x) - F,
            sym.diff(sym.diff(L,dt1),t) - sym.diff(L,t1),
            sym.diff(sym.diff(L,dt2),t) - sym.diff(L,t2)
        ]
    G = {k:v.simplify() for k,v in sym.solve(EOM, [ddx,ddt1,ddt2]).items()}
    print("Equations of motion:")
    sym.pprint(G)

    ###############################################################################################
    ######################################## Part B ###############################################
    ###############################################################################################
    # construct Jacobian
    J_orig = sym.Matrix([
            [sym.diff(G[ddx],v) for v in vars_],
            [sym.diff(G[ddt1],v) for v in vars_],
            [sym.diff(G[ddt2],v) for v in vars_]
        ])

    # substitute our origin conditions (x,dx,t1,dt1,t2,dt2) = 0
    J = J_orig.subs([(v,0) for v in vars_])
    print("Jacobian at origin:")
    sym.pprint(J)
    
    # Convert to our state space representation
    zeros = sym.zeros(1,6)
    A = sym.Matrix([zeros, J[0,:], zeros, J[1,:], zeros, J[2,:]])
    A[0,1] = A[2,3] = A[4,5] = sym.Rational(1)
    B = sym.Matrix([0,1/M,0,1/(l1*M),0,1/(l2*M)])
    print("State Space Representation (A,B):")
    sym.pprint(A)
    sym.pprint(B)

    ###############################################################################################
    ######################################## Part C ###############################################
    ###############################################################################################
    # Determine Controllability Conditions
    C = B
    for i in range(1,6):
        C = C.row_join(A**i * B)
    sym.pprint(C)

    # Controllability conditions:
    print(C.det())

    ###############################################################################################
    ######################################## Part D ###############################################
    ###############################################################################################
    # check conditions in part D
    vals = [(g,9.81),(M,1000),(m1,100),(m2,100),(l1,20),(l2,10)]
    C_subs = C.subs(vals)

    # sanity check rank
    assert(C_subs.rank() == 6)
    print("Part D: Rank {}".format(C_subs.rank()))
    sym.pprint(C_subs)

    # update A,B matrices with these values
    A = np.array(A.subs(vals),dtype=np.float32)
    B = np.array(B.subs(vals),dtype=np.float32)

    # solve the Riccati equation:
    P = solve_continuous_are(A, B, Q, R)
    K = (B.T@P)/R

    # plot the response to an initial offset from 0

    # closed loop system
    SYS_CL = scipy.signal.StateSpace(A-B*K,B,np.eye(6))

    # simulated response
    T,Y,X = scipy.signal.lsim(SYS_CL, None, Times, IC)

    # plot response
    plt.figure("Closed Loop Response")
    plt.plot(T,X[:,0], 'b')
    plt.plot(T,X[:,2], 'r')
    plt.plot(T,X[:,4], 'k')

    # simulate the nonlinear system
    G_subs = {k:v.subs(vals).subs(F,0).evalf() for k,v in G.items()}
    def ODE(time, y):
        state = [s for s in zip([x,dx,t1,dt1,t2,dt2], y)]
        command = (B*K)@np.array(y)
        result = [
            y[1],
            (G_subs[ddx].subs(state).subs(t,time) - command[1]).simplify(),
            y[3],
            (G_subs[ddt1].subs(state).subs(t,time) - command[3]).simplify(),
            y[5],
            (G_subs[ddt2].subs(state).subs(t,time) - command[5]).simplify()
        ]
        return result

    # integrate
    nonlinear = scipy.integrate.solve_ivp(ODE, [Times[0],Times[-1]], IC)
    assert(nonlinear.success)

    # plot the results
    plt.plot(nonlinear.t, nonlinear.y[0,:], '--b')
    plt.plot(nonlinear.t, nonlinear.y[2,:], '--r')
    plt.plot(nonlinear.t, nonlinear.y[4,:], '--k')

    plt.grid(True)
    plt.legend(["Linear X","Linear Theta1","Linear Theta2", "Nonlinear X", "Nonlinear Theta1", "Nonlinear Theta2"])
    plt.show()

    ###############################################################################################
    ##################################### Interaction #############################################
    ###############################################################################################

    import code
    code.interact(local=locals())

