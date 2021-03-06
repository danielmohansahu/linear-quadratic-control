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
constant_values = [(g,9.81),(M,1000),(m1,100),(m2,100),(l1,20),(l2,10)]

# core state variables
t = sym.Symbol("t")
x = sym.Function("x")(t)
dx = x.diff(t)
t1 = sym.Function("theta1")(t)
dt1 = t1.diff(t)
t2 = sym.Function("theta2")(t)
dt2 = t2.diff(t)
states = [x,dx,t1,dt1,t2,dt2]

# other variables
ddx = dx.diff(t)
ddt1 = dt1.diff(t)
ddt2 = dt2.diff(t)
F = sym.Symbol("F")

# Energy Equations (basic inputs)
T = 0.5 * (M + m1 + m2)*dx**2 - m1*l1*dx*dt1*sym.cos(t1) + 0.5*m1*l1**2*dt1**2 - m2*l2*dx*dt2*sym.cos(t2) + 0.5*m2*l2**2*dt2**2
V = -m1*g*l1*sym.cos(t1) - m2*g*l2*sym.cos(t2)

# Simulation and Control Parameters
Times = np.arange(0,60,1e-3)
IC = np.array([0,1e-3,0,5e-5,0,1e-4])
# Q = np.diag([1,100,1,10,1,10])
# R = 0.001
Q = np.diag([1,0.5,1000,500,1000,500])
R = 1.0e-4
STEP = 0.1

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
        [sym.diff(G[ddx],v) for v in states],
        [sym.diff(G[ddt1],v) for v in states],
        [sym.diff(G[ddt2],v) for v in states]
    ])

    # substitute our origin conditions (x,dx,t1,dt1,t2,dt2) = 0
    J = J_orig.subs([(v,0) for v in states]).doit()
    
    print("Jacobian at origin:")
    sym.pprint(J)
    
    # Convert to our linearized state space representation
    A = sym.Matrix([sym.zeros(1,6), J[0,:], sym.zeros(1,6), J[1,:], sym.zeros(1,6), J[2,:]])
    A[0,1] = A[2,3] = A[4,5] = sym.Rational(1)
    B = sym.Matrix([0,1/M,0,1/(l1*M),0,1/(l2*M)])

    print("State Space Representation (A,B):")
    sym.pprint(A)
    sym.pprint(B)

    ###############################################################################################
    ######################################## Part C ###############################################
    ###############################################################################################
    # Determine Controllability Conditions
    Controllability = B
    for i in range(1,6):
        Controllability = Controllability.row_join(A**i * B)

    print("Controllability matrix: ")
    sym.pprint(Controllability)

    # Controllability conditions:
    print("System is controllable iff the following is satisfied: ")
    print(Controllability.det())

    ###############################################################################################
    ######################################## Part D ###############################################
    ###############################################################################################
    # check conditions in part D
    Controllability_subs = Controllability.subs(constant_values)

    # sanity check rank
    assert(Controllability_subs.rank() == 6)

    print("Part D: Rank {}".format(Controllability_subs.rank()))
    sym.pprint(Controllability_subs)

    # update A,B matrices with these values
    A = np.array(A.subs(constant_values),dtype=np.float32)
    B = np.array(B.subs(constant_values),dtype=np.float32)

    # solve the Riccati equation:
    P = solve_continuous_are(A, B, Q, R)
    K = (B.T@P)/R

    # plot the response to an initial offset from 0

    # closed loop system
    SYS_CL = scipy.signal.StateSpace(A-B*K,B,np.eye(6))

    # simulate the nonlinear system
    G_subs = {k:sym.lambdify(states,v.subs(constant_values).subs(F,0).evalf(),"numpy") for k,v in G.items()}
    def ODE(time, y):
        state = [
            y[1],
            G_subs[ddx](*y),
            y[3],
            G_subs[ddt1](*y),
            y[5],
            G_subs[ddt2](*y)
        ]
        command = (B*K)@np.array(y)
        return state - command

    # integrate
    T,Y,X = scipy.signal.lsim(SYS_CL, None, Times, IC)

    # plot response
    plt.figure("Closed Loop Response")
    plt.plot(T,X[:,0], 'b')
    plt.plot(T,X[:,2], 'r')
    plt.plot(T,X[:,4], 'k')

    nonlinear = scipy.integrate.solve_ivp(ODE, [Times[0],Times[-1]], IC)
    assert(nonlinear.success)

    # plot the results
    plt.plot(nonlinear.t, nonlinear.y[0,:], '--b')
    plt.plot(nonlinear.t, nonlinear.y[2,:], '--r')
    plt.plot(nonlinear.t, nonlinear.y[4,:], '--k')

    plt.grid(True)
    plt.legend(["Linear X","Linear Theta1","Linear Theta2", "Nonlinear X", "Nonlinear Theta1", "Nonlinear Theta2"])
    plt.show()

    # check that eigenvalues of the linearized closed loop system are all in the LHP
    eigenvalues = np.linalg.eig(A-B*K)[0]
    assert(np.all(eigenvalues < 0))
    print("The linearized closed loop system is locally asymptotically stable.")

    ###############################################################################################
    ######################################## Part E ###############################################
    ###############################################################################################

    # examine the observability of various different C matrices
    Potentials = [
        sym.Matrix([
            [1,0,0,0,0,0],  # x
        ]),
        sym.Matrix([
            [0,0,1,0,0,0],  # t1
            [0,0,0,0,1,0]   # t2
        ]),
        sym.Matrix([
            [1,0,0,0,0,0],  # x
            [0,0,0,0,1,0]   # t2
        ]),
        sym.Matrix([
            [1,0,0,0,0,0],  # x
            [0,0,1,0,0,0],  # t1
            [0,0,0,0,1,0]   # t2
        ])
    ]
    Observables = []
    for i,c in enumerate(Potentials):
        # check determinant of full system
        A_temp = sym.Matrix(A)

        obs = sym.Matrix(c)
        for j in range(1,6):
            obs = obs.col_join(c*(A_temp)**j)
        
        if obs.rank() == 6:
            print("Potential C matrix #{} is observable. Rank: {}".format(i+1, obs.rank()))
            Observables.append(sym.Matrix(c))
        else:
            print("Potential C matrix #{} is NOT observable. Rank: {}".format(i+1, obs.rank()))

    ###############################################################################################
    ######################################## Part F ###############################################
    ###############################################################################################

    # construct desired poles (order of magnitude larger than controller eigenvalues)
    observer_poles = np.array([np.complex(10*sym.re(eig), sym.im(eig)) for eig in eigenvalues])

    # obtain observers for each output vector
    for i,C in enumerate(Observables):
        # initialize plots
        fig = plt.figure("Closed Loop Observer Response C_{}".format(i+1))

        # place poles of A-LC an order of magnitude farther left than the controller poles
        C = np.array(C, dtype=np.float32)
        L = scipy.signal.place_poles(A.T, C.T, observer_poles).gain_matrix.T
        print("Found observer for potential C matrix #{}".format(i+1))

        # simulate the linear response
        Ao = np.block([[A-B@K, B@K],[np.zeros(A.shape),A-L@C]])
        Bo = np.block([[B],[np.zeros(B.shape)]])
        Co = np.block([[C, np.zeros(C.shape)]])
        SYS_CLO = scipy.signal.StateSpace(Ao,Bo,Co)

        # initial condition response
        Tic,Yic,Xic = scipy.signal.lsim(SYS_CLO, None, Times, np.hstack((IC,IC)))

        # unit step response
        U = np.ones(Times.shape)*STEP
        U[:1000] *= 0
        Ts,Ys,Xs = scipy.signal.lsim(SYS_CLO, U, Times, np.hstack((np.zeros(IC.shape),np.zeros(IC.shape))))

        # estimation is the difference between actual state and estimation error
        Xic_est = Xic[:,:6]-Xic[:,6:]
        Xs_est = Xs[:,:6]-Xs[:,6:]

        # plot linear initial condition response
        plt.subplot(221)
        plt.title("Linear IC Response")
        plt.plot(Tic,Xic[:,0], 'b')
        plt.plot(Tic,Xic_est[:,0], '--b')
        plt.plot(Tic,Xic[:,2], 'r')
        plt.plot(Tic,Xic_est[:,2], '--r')
        plt.plot(Tic,Xic[:,4], 'k')
        plt.plot(Tic,Xic_est[:,4], '--k')
        plt.grid(True)
        plt.legend(["X","X_obs","theta1","theta1_obs","theta2","theta2_obs"],loc=1)
        plt.xlim(T[0],T[-1])

        # plot linear step response
        plt.subplot(223)
        plt.title("Linear Step Response")
        plt.plot(Ts,Xs[:,0], 'b')
        plt.plot(Ts,Xs_est[:,0], '--b')
        plt.plot(Ts,Xs[:,2], 'r')
        plt.plot(Ts,Xs_est[:,2], '--r')
        plt.plot(Ts,Xs[:,4], 'k')
        plt.plot(Ts,Xs_est[:,4], '--k')
        plt.grid(True)
        plt.legend(["X","X_obs","theta1","theta1_obs","theta2","theta2_obs"],loc=1)
        plt.xlim(T[0],T[-1])

        # simulate the nonlinear response
        def ODE(time, y, step_time):
            state = [
                y[1],
                G_subs[ddx](*y[:6]),
                y[3],
                G_subs[ddt1](*y[:6]),
                y[5],
                G_subs[ddt2](*y[:6]),
                y[7],
                G_subs[ddx](*y[6:]),
                y[9],
                G_subs[ddt1](*y[6:]),
                y[11],
                G_subs[ddt2](*y[6:])
            ]
            command = (B@K)@(np.array(y[6:])-np.array(y[:6]))
            if time >= step_time:
                command += (B*STEP).reshape(command.shape)
            observe = (L@C)@np.array(y[6:])
            return state + np.hstack((command, -observe))

        # response to initial conditions
        resp_ic = scipy.integrate.solve_ivp(ODE, [Times[0],Times[-1]], np.hstack((IC,IC)), args=(np.inf,))
        assert(resp_ic.success)

        # response to step input
        resp_s = scipy.integrate.solve_ivp(ODE, [Times[0],Times[-1]], np.hstack((np.zeros(IC.shape),np.zeros(IC.shape))), args=(1.0,))
        assert(resp_s.success)

        # plot the initial condition results
        plt.subplot(222)
        plt.title("Nonlinear IC Response")
        plt.plot(resp_ic.t, resp_ic.y[0,:], 'b')
        plt.plot(resp_ic.t, resp_ic.y[0,:]-resp_ic.y[6,:], '--b')
        plt.plot(resp_ic.t, resp_ic.y[2,:], 'r')
        plt.plot(resp_ic.t, resp_ic.y[2,:]-resp_ic.y[8,:], '--r')
        plt.plot(resp_ic.t, resp_ic.y[4,:], 'k')
        plt.plot(resp_ic.t, resp_ic.y[4,:]-resp_ic.y[10,:], '--k')
        plt.grid(True)
        plt.legend(["X","X_obs","theta1","theta1_obs","theta2","theta2_obs"],loc=1)
        plt.xlim(T[0],T[-1])

        # plot the step response results
        plt.subplot(224)
        plt.title("Nonlinear Step Response")
        plt.plot(resp_s.t, resp_s.y[0,:], 'b')
        plt.plot(resp_s.t, resp_s.y[0,:]-resp_s.y[6,:], '--b')
        plt.plot(resp_s.t, resp_s.y[2,:], 'r')
        plt.plot(resp_s.t, resp_s.y[2,:]-resp_s.y[8,:], '--r')
        plt.plot(resp_s.t, resp_s.y[4,:], 'k')
        plt.plot(resp_s.t, resp_s.y[4,:]-resp_s.y[10,:], '--k')
        plt.grid(True)
        plt.legend(["X","X_obs","theta1","theta1_obs","theta2","theta2_obs"],loc=1)
        plt.xlim(T[0],T[-1])


        # plot an extra large version of the IC response
        resp_lqg = scipy.integrate.solve_ivp(ODE, [Times[0],Times[-1]], np.hstack((IC*10,np.zeros(IC.shape))), args=(np.inf,))
        plt.figure("LQG Response (C_{})".format(i+1))
        plt.title("LQG Initial Condition Response")
        plt.plot(resp_lqg.t, resp_lqg.y[0,:], 'b')
        plt.plot(resp_lqg.t, resp_lqg.y[0,:]-resp_lqg.y[6,:], '--b')
        plt.plot(resp_lqg.t, resp_lqg.y[2,:], 'r')
        plt.plot(resp_lqg.t, resp_lqg.y[2,:]-resp_lqg.y[8,:], '--r')
        plt.plot(resp_lqg.t, resp_lqg.y[4,:], 'k')
        plt.plot(resp_lqg.t, resp_lqg.y[4,:]-resp_lqg.y[10,:], '--k')
        plt.grid(True)
        plt.legend(["X","X_obs","theta1","theta1_obs","theta2","theta2_obs"])
        plt.xlim(T[0],T[-1])

    # plot
    plt.show()

    ###############################################################################################
    ######################################## Part G ###############################################
    ###############################################################################################

    # uses a similar plot to the ones above

    ###############################################################################################
    ##################################### Interaction #############################################
    ###############################################################################################

    import code
    code.interact(local=locals())

