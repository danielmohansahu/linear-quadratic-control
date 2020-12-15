clc
clear all
%%%PART 1%%%
syms th1 th2 L1 L2  M m1 m2 F x xd xdd th1d th1dd th2d th2dd g
%%%%%Kinematics%%%%
KEM = .5*M*xd^2;
KEm1 = .5*m1*((xd - L1*th1d*cos(th1))^2 + (L1*th1d*sin(th1))^2);
KEm2 = .5*m2*((xd - L2*th2d*cos(th2))^2 + (L2*th2d*sin(th2))^2);
PE = -1*(m1*g*L1*cos(th1) + m2*g*L2*cos(th2));
KE = simplify(KEM + KEm1 + KEm2)
L = simplify(KE - PE)
%%%%EL second term%%%%
dLx = simplify(diff(L,x))
dLth1 = simplify(diff(L,th1))
dLth2 = simplify(diff(L,th2))
%%%%EL first term part 1%%%%
dLxd = simplify(diff(L,xd))
dLth1d = simplify(diff(L,th1d))
dLth2d = simplify(diff(L,th2d))
%%%%EL first term part2%%%%
ddLxd = simplify(M*xdd + m1*(xdd - L1*th1dd*cos(th1) + L1*th1d^2*sin(th1)) + m2*(xdd - L2*th2dd*cos(th2) + L2*th2d^2*sin(th2)));
ddLth1d = simplify(L1*m1*(L1*th1dd - xdd*cos(th1) + xd*th1d*sin(th1)));
ddLth2d = simplify(L2*m2*(L2*th2dd - xdd*cos(th2) + xd*th2d*sin(th2)));
%%%%EoM for the system%%%%
LEx = simplify(ddLxd - dLx)
LEth1 = simplify(ddLth1d - dLth1)
LEth2 = simplify(ddLth2d - dLth2)
%xdd = (F + m1*L1*th1dd*cos(th1) + m2*L2*th2dd*cos(th2) -m1*L1*sin(th1)*th1d^2 - m2*L2*sin(th2)*th2d^2)/(m1+m2+M);
%th1dd  = -((g*sin(th1))/L1) + (xdd*cos(th1))/L1;
%th2dd  = -((g*sin(th2))/L2) + (xdd*cos(th2))/L2;
%%%%Non-linear State-Spece%%%%
syms x2 x4 x6 x4d x6d x2d x1 xd x3 th1 x4 th1d x5 th2 x6 th2d
th1dd  = -((g*sin(th1))/L1) + (xdd*cos(th1))/L1;
th2dd  = -((g*sin(th2))/L2) + (xdd*cos(th2))/L2;
solve_xdd = simplify((F + m1*L1*th1dd*cos(th1) + m2*L2*th2dd*cos(th2) -m1*L1*sin(th1)*th1d^2 - m2*L2*sin(th2)*th2d^2))

only_xdd=(- L1*m1*sin(th1)*th1d^2 - L2*m2*sin(th2)*th2d^2 - g*m1*sin(th1)*cos(th1) - g*m2*sin(th2)*cos(th2) + F)/(-m1*cos(th1)^2 - m2*cos(th2)^2 + M + m1 + m2);
only_th1dd  = simplify(-((g*sin(th1))/L1) + (only_xdd*cos(th1))/L1)
only_th2dd  = simplify(-((g*sin(th2))/L2) + (only_xdd*cos(th2))/L2)

x1 = x;
x1d = x2;
x2 = xd;
x2d = only_xdd;
x3 = th1;
x3d = x4;
x4 = th1d;
x4d = only_th1dd;
x5 = th2;
x5d = x6;
x6 = th2d;
x6d = only_th2dd;

Xd = [x1d;x2d;x3d;x4d;x5d;x6d;]
%%%%Equilibrium Points%%%%
syms x1 x2 x3 x4 x5 x6
Xd = subs(Xd, th1, x3);
Xd = subs(Xd, th2, x5);
Xd = subs(Xd, th1d, x4);
Xd = subs(Xd, th2d, x6);
Xd = subs(Xd, xdd, x2d)

Jx = jacobian(Xd, [x1, x2, x3, x4, x5, x6])
Ju = jacobian(Xd,[F])
Jx = subs(Jx,x1,0);
Jx = subs(Jx,x3,0);
Jx = subs(Jx,x4,0);
Jx = subs(Jx,x5,0);
AF =subs(Jx,x6,0)
Ju = subs(Ju,x1,0);
Ju = subs(Ju,x3,0);
Ju = subs(Ju,x4,0);
Ju = subs(Ju,x5,0);
BF = subs(Ju,x6,0)
%%%%Controllablity Conditions%%%%%
AB = AF*BF;
A2B = AF^2*BF;
A3B = AF^3*BF;
A4B = AF^4*BF;
A5B = AF^5*BF;
M = [BF AB A2B A3B A4B A5B];
rM = rank(M);
dM = simplify(det(M));
%Controllable if L1 != L2
%%%%Check Controllablity with values%%%%
M = 1000;
m1 = 100;
m2 = 100;
g = 10;
L1 = 20;
L2 = 10;
AF = [0, 1, 0, 0, 0, 0; 0, 0, -(g*m1)/M, 0, -(g*m2)/M, 0;0, 0, 0, 1, 0, 0;0, 0, -g/L1-(g*m1)/(L1*M), 0, -(g*m2)/(L1*M), 0;0, 0, 0, 0, 0, 1;0, 0,  -(g*m1)/(L2*M), 0, -g/L2-(g*m2)/(L2*M), 0]
BF = [0; 1/M; 0; 1/(L1*M); 0; 1/(L2*M)]
C = [1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0];
Co = ctrb(AF,BF);
unco = length(AF) - rank(Co)
%The system is controllable
%%%%%LQR Controller%%%%%
R = .1*1.0e-03
Q = [1 0 0 0 0 0;
     0 .5 0 0 0 0;
     0 0 1000 0 0 0;
     0 0 0 500 0 0;
     0 0 0 0 1000 0;
     0 0 0 0 0 500]
K = lqr(AF,BF,Q,R)

Ac = [(AF-BF*K)]
Bc = [BF]
Cc = [C];
Dc = [0;0;0];
x0 = [.1 0 .1 0 .1 0];
t = 0:0.1:70;
u = zeros(size(t));
[y,x] =lsim(Ac,Bc,Cc,Dc,u,t,x0);
figure(1)
plot(t,y)
%%%%Nonlinear Reponse%%%%
%x0 = [0.1, 0.1, 0.1];
%[T,X] = ode45(@pen,[0,20],[30,0]);
%plot(T,X(:,1),'-',T,X(:,2),'-',T,X(:,3),'-',T,X(:,4),'-',T,X(:,5),'-',T,X(:,6),'-')
%function dx = pen(~,x)
%dx = zeros(6,1); % column vector
%dx(1) = x(2);
%dx(2) = -(20*100*sin(x(3))*x(4)^2 + 10*100*sin(x(5))*x(6)^2 - F + 10*100*cos(x(3))*sin(x(3)) + 10*100*cos(x(5))*sin(x(5)))/(-100*cos(x(3))^2 - 100*cos(x(5))^2 + 1000 +100 + 100);
%dx(3) = x(4)
%dx(4) = -(10*sin(x(3)))/20 - (cos(x(3))*(20*100*sin(x(3))*x(4)^2 + 10*100*sin(x(5))*x(6)^2 - F + 10*100*cos(x(3))*sin(x(3)) + 10*100*cos(x(5))*sin(x(5))))/(20*(- 100*cos(x(5))^2 - 100*cos(x(5))^2 + 1000 + 100 + 100))
%dx(5) = x(6)
%dx(6) = -(10*sin(x(5)))/10 - (cos(x(5))*(20*100*sin(x(3))*x(4)^2 + 10*100*sin(x(5))*x(6)^2 - F + 10*100*cos(x(3))*sin(x(3)) + 10*100*cos(x(5))*sin(x(5))))/(10*(- 100*cos(x(3))^2 - 100*cos(x(6))^2 + 1000 + 100 + 100))
%end
%%%%%Lyapunov Stability of CL System%%%%%
Ac_Eig = eig(Ac)
% All Eigenvalues are negative so the CL system is at least locally stable
%%%PART 2%%%
%%%%Observability%%%%%
Cx = [1 0 0 0 0 0];
Cth12 = [0 0 1 0 1 0];
Cxth2 = [1 0 0 0 1 0];
Obx = obsv(Ac,Cx);
Obth12 = obsv(Ac,Cth12);
Obxth2 = obsv(Ac,Cxth2);
Ob = obsv(Ac,C);
unobx = length(Ac)-rank(Obx)
unobth12 = length(Ac)-rank(Obth12)
unobxth2 = length(Ac)-rank(Obxth2)
unob = length(Ac)-rank(Ob)
% The linearized system is observable only for x,th1,th2
%%%%%Luenberger Observer%%%%%
%Lx = place(Ac',Cx',P)'
%Acex = (Ac-L*Cx)
