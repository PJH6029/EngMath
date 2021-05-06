import numpy as np
from RK import RK
from Utils import Utils
from Euler import Euler
from Iteration import Iteration
from PDE import PDE
np.set_printoptions(precision=12)
# print(2 * pow(np.e, -0.5*0.1) + pow(np.e, -1.5 * 0.1))
# print(-1 * pow(np.e, -0.5*0.1) - 1.5 * pow(np.e, -1.5*0.1))


if __name__ == '__main__':
    # email example (first order)
    # y' = (y-x-1)^2 + 2, y0 = 1, h = 0.1

    # input 형식(f, xn, yn, h)
    def f(x, y):  # y' = f(x, y)
        return pow(y - x - 1, 2) + 2
    x0 = 0
    y0 = 1
    h = 0.1

    # 사용법
    y1_Euler = Euler.eulerMethod(f, x0, y0, h)
    y1_improvedEuler = Euler.improvedEulerMethod(f, x0, y0, h)
    y1_RKClassic = RK.RKClassic(f, x0, y0, h)
    y1_4th = RK.RKFourth(f, x0, y0, h)  # 4-th order
    y1_5th = RK.RKFifth(f, x0, y0, h)  # 5-th order

    error = y1_5th - y1_4th  # email pdf에 나와있는 에러

    # 결과 출력법
    # Utils.printResult(f, x0, y0, h, exact_y=1.20033467209)  # pdf에 있는 실제 y값 사용했음
    # Utils.printResult(f, x0, y0, h)  # exact_y를 주지 않았을 때의 출력


    # -----------------------------------------------------------------------------------------
    # pdf system example
    # y'' + 2y' + 0.75y = 0, y0 = 3, y'0 = -2.5, h = 0.2

    # input 형식(F, xn, yn, h), y_(n)prime = F(x, *y_vec)
    def F(x, *y_vec):  # y'' = -0.75y -2y'
        return -0.75 * y_vec[0] - 2 * y_vec[1]
    xn = 0
    yn_vec = np.array([3, -2.5])
    h = 0.2

    # 사용법
    y1_Euler_vec = Euler.eulerMethodForSystem(scalarF=F, xn=xn, yn_vec=yn_vec, h=h)
    y1_improvedEuler_vec = Euler.improvedEulerMethodForSystem(scalarF=F, xn=xn, yn_vec=yn_vec, h=h)
    y1_vec_classicRK = RK.RKClassicForSystem(scalarF=F, xn=xn, yn_vec=yn_vec, h=h)
    y1_4th_vec = RK.RKFourthForSystem(scalarF=F, xn=xn, yn_vec=yn_vec, h=h)
    y1_5th_vec = RK.RKFifthForSystem(scalarF=F, xn=xn, yn_vec=yn_vec, h=h)

    # 실제 해: 2 * e^(-0.5x) + e^(-1.5x)
    y = 2 * pow(np.e, -0.5 * 0.2) + pow(np.e, -1.5 * 0.2)
    yprime = pow(np.e, -0.5 * 0.2) * (-1) + (-3 / 2) * pow(np.e, -1.5 * 0.2)
    exact_y_vec = [y, yprime]

    # 결과 출력법
    # Utils.printResultForSystem(F, xn, yn_vec, h, exact_y_vec=exact_y_vec)
    # Utils.printResultForSystem(F, xn, yn_vec, h)  # exact_y_vec을 주지 않았을 때의 출력


    # -----------------------------------------------------------------------------------------
    # target error보다 작은 에러를 얻기 위한 h구하는 함수. default stride = 0.05 (근데 많이 안해봄)
    h_target = Utils.get_h_for_target_error(exact_y=1.20033467209, method=RK.RKFifth,
                                               target_error=3.04e-9, f=f, xn=x0, yn=y0, stride=0.1)
    # print(f"h target: {h_target}")


    # -----------------------------------------------------------------------------------------
    # Iteration example

    # input 형식(matrix A, vector b) -> Ax = b 근사
    A = np.array([[-4, 1, 1, 0],
                  [1, -4, 0, 1],
                  [2, 0, -4, 1],
                  [0, 2, 1, -4]])

    b_vec = np.array([0.75, 1.125, -1.5, -6])

    # 사용법
    x_GS_vec = Iteration.GaussSeidelMethod(A, b_vec)
    x_J_vec = Iteration.JacobiMethod(A, b_vec)

    # 결과 출력법
    # Utils.printResultOfIteration(coeff_mat=A, b_vec=b_vec)

    # TODO n번째 yn구하기, vector input validation(list -> np.ndarray)

    # -----------------------------------------------------------------------------------------
    # PDE(elliptic example)

    # input 형식:
    # (xrange, yrange, h, uL=None, uR=None, uB=None, uT=None, f=None,
    #                     un_axis=None, un=None,  # for mixed boundary condition
    #                     boundary_type='Dirichlet')

    # boundary_type: 'dirichlet'(default) or ['mixed' or 'neumann']. 대소문자 관계 x, mixed == neumann
    # uL, uR, uB, uT: 함수( of x, y)이거나 상수
    # f: 기본값 None(Laplace), f(x,y) 넣어주면 포아송 방정식
    # un_axis: un에서 n축, un: x축이면 function of y, y축이면 function of x

    # ex1 laplace, dirichlet
    Laplace_Dirichlet = PDE.PoissonEquation(xrange=(0, 12), yrange=(0, 12), h=4,
                                            uL=0, uR=100, uB=100, uT=0)
    print("Laplace and Dirchlet:")
    print(Laplace_Dirichlet)
    print()

    # ex2 poisson, neumann(y-axis)
    def f(x, y):
        return 12 * x * y
    def uR(x, y):
        return 3 * pow(y, 3)
    def un(x):
        return 6 * x

    Poisson_Neumann = PDE.PoissonEquation(xrange=(0, 1.5), yrange=(0, 1.0), h=0.5,
                                          uL=0, uR=uR, uB=0, f=f,
                                          un_axis='y', un=un,
                                          boundary_type='mixed')
    print("Poisson and Neumann(y):")
    print(Poisson_Neumann)
    print()


    # ex2 poisson, neumann(x-axis)
    def f(x, y):
        return pow(x, 2) + pow(y, 2)
    def uT(x, y):
        return 9 * pow(x, 2)
    def un(y):
        return 6 * pow(y, 2)

    Poisson_Neumann = PDE.PoissonEquation(xrange=(0, 3), yrange=(0, 3), h=1,
                                          uL=0, uB=0, uT=uT, f=f,
                                          un_axis='x', un=un,
                                          boundary_type='mixed')
    print("Poisson and Neumann(x):")
    print(Poisson_Neumann)
    print()