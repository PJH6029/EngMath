import numpy as np
from ode.RK import RK
from ode.Euler import Euler
from iteration.Iteration import Iteration
from utils.Utils import Utils

np.set_printoptions(precision=12)

if __name__ == '__main__':
    # email example (first order)
    # y' = (y-x-1)^2 + 2, y0 = 1, h = 0.1

    # input 형식(f, xn, yn, h)
    def f(x, y):  # y' = f(x, y)
        return 2 * x * y


    x0 = 1
    y0 = 1
    h = 0.1

    for _ in range(10):
        y0 = RK.RKClassic(f, x0, y0, h)
        x0 = x0 + h

        print(f"({x0:.1f}, {y0:.10f})zW")

    # 사용법
    y1_Euler = Euler.eulerMethod(f, x0, y0, h)
    y1_improvedEuler = Euler.improvedEulerMethod(f, x0, y0, h)
    y1_RKClassic = RK.RKClassic(f, x0, y0, h)
    y1_4th = RK.RKFourth(f, x0, y0, h)  # 4-th order
    y1_5th = RK.RKFifth(f, x0, y0, h)  # 5-th order

    ## error = y1_5th - y1_4th  # email pdf에 나와있는 에러

    # 결과 출력법
    ## Utils.printResult(f, x0, y0, h, exact_y=1.20033467209)  # pdf에 있는 실제 y값 사용했음
    ## Utils.printResult(f, x0, y0, h)  # exact_y를 주지 않았을 때의 출력

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
    ## Utils.printResultForSystem(F, xn, yn_vec, h, exact_y_vec=exact_y_vec)
    ## Utils.printResultForSystem(F, xn, yn_vec, h)  # exact_y_vec을 주지 않았을 때의 출력

    # -----------------------------------------------------------------------------------------
    # target error보다 작은 에러를 얻기 위한 h구하는 함수. default stride = 0.05 (근데 많이 안해봄)
    h_target = Utils.get_h_for_target_error(exact_y=1.20033467209, method=RK.RKFifth,
                                            target_error=3.04e-9, f=f, xn=x0, yn=y0, stride=0.1)
    ## print(f"h target: {h_target}")

    # -----------------------------------------------------------------------------------------
    # Iteration example

    # input 형식(matrix A, vector b) -> Ax = b 근사
    A = np.array([[-4, 1, 1, 0],
                  [1, -4, 0, 1],
                  [1, 0, -4, 1],
                  [0, 1, 1, -4]])

    b_vec = np.array([-200, -200, -100, -100])

    # 사용법
    x_GS_vec = Iteration.GaussSeidelMethod(A, b_vec)
    x_J_vec = Iteration.JacobiMethod(A, b_vec)

    # 결과 출력법
    ## Utils.printResultOfIteration(coeff_mat=A, b_vec=b_vec)

    # TODO n번째 yn구하기, vector input validation(list -> np.ndarray)
