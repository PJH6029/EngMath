import numpy as np
import inspect
from ode.Euler import Euler
from ode.RK import RK

np.set_printoptions(precision=12)


class Utils:
    @staticmethod
    def get_h_for_target_error(exact_y, method, target_error, f, xn, yn, stride=0.05):
        for h in np.arange(1, 0, -stride):
            y_expected = method(f, xn, yn, h)
            error = exact_y - y_expected
            if abs(error) < target_error:
                return h
        return -1

    @staticmethod
    def get_F_from_ODE(F):
        # scalarF -> n차원 vec F(n: 미분 order)
        def returnF(x, *y_vec):  # y_vec: y~y_n-1prime, F(x, *y_vec)
            return np.array([
                *y_vec[1:],
                F(x, *y_vec)
            ])

        return returnF

    @staticmethod
    def get_real_error(exact_y, approximated_y):
        return exact_y - approximated_y

    @staticmethod
    def get_real_error_vec(exact_y_vec, approximated_y_vec):
        return np.array([exact_y_vec[i] - approximated_y_vec[i] for i in range(len(exact_y_vec))])

    @staticmethod
    def printResult(f, xn, yn, h, exact_y=None):
        method_arr = [Euler.eulerMethod, Euler.improvedEulerMethod,
                      RK.RKClassic, RK.RKFourth, RK.RKFifth]

        y_next_arr = [method_arr[i](f, xn, yn, h) for i in range(len(method_arr))]
        error_arr = list()
        if exact_y:
            error_arr = [Utils.get_real_error(exact_y, y_next_arr[i]) for i in range(len(y_next_arr))]

        if method_arr:
            nameList = [method_arr[i].__name__ for i in range(len(method_arr))]
            maxNameLength = max([len(nameList[i]) for i in range(len(nameList))])
            maxNameLength = max(len("Method"), maxNameLength)
        else:
            nameList = list()
            maxNameLength = len("Method")

        if exact_y:
            maxLength = maxNameLength + 59
        else:
            maxLength = maxNameLength + 20

        funcSource = inspect.getsource(f)
        funcLines = funcSource.split('\n')
        maxTab = len(funcLines[0]) - len(funcLines[0].lstrip())
        for i, line in enumerate(funcLines):
            funcLines[i] = line[maxTab:]
            maxLength = max(maxLength, len(line[maxTab:]) + 10)

        print()
        print(f"{'Result of Approximation':-^{maxLength}}")
        print("Input parameter:")
        for i, line in enumerate(funcLines):
            if i == 0:
                print("function: " + line)
            else:
                print(" " * 10 + line)
        print(f"{'nth x':>8}: {xn}")
        print(f"{'nth y':>8}: {yn}")
        print(f"{'h':>8}: {h}")
        print()

        print("Output:")
        print(f"{'Method':^{maxNameLength}}      {'approximated y':^15}", end='')
        if exact_y:
            print(f"    {'error':^15}    {'real y':^15}")
        else:
            print()
        print('-' * maxLength)

        for i in range(len(method_arr)):
            print(f"{nameList[i]:^{maxNameLength}} :    {y_next_arr[i]:+12.12f}", end='')
            if exact_y:
                print(f"    {error_arr[i]:+12.12f}", end='')
                if i == 0:
                    print(f"    {exact_y:+12.12f}")
                else:
                    st = "''"
                    print(f'    {st:^15}')
            else:
                print()
            print()

        print(f"{'Print finished':-^{maxLength}}")
        print()

        '''
        y_next_Euler = Euler.eulerMethod(f, xn, yn, h)
        y_next_improvedEuler = Euler.improvedEulerMethod(f, xn, yn, h)
        y_next_RKClassic = RK.RKClassic(f, xn, yn, h)
        y_next_4th = RK.RKFourth(f, xn, yn, h)
        y_next_5th = RK.RKFifth(f, xn, yn, h)
        '''

    @staticmethod
    def printResultForSystem(scalarF, xn, yn_vec, h, exact_y_vec=None):
        method_arr = [Euler.eulerMethodForSystem, Euler.improvedEulerMethodForSystem,
                      RK.RKClassicForSystem, RK.RKFourthForSystem, RK.RKFifthForSystem]

        y_next_arr_vec = [method_arr[i](scalarF, xn, yn_vec, h) for i in range(len(method_arr))]
        error_arr_vec = np.array([])
        if exact_y_vec is not None:
            if type(exact_y_vec) != type(error_arr_vec):
                exact_y_vec = np.array(exact_y_vec)
            error_arr_vec = [Utils.get_real_error_vec(exact_y_vec, y_next_arr_vec[i]) for i in
                             range(len(y_next_arr_vec))]

        if method_arr:
            nameList = [method_arr[i].__name__ for i in range(len(method_arr))]
            maxNameLength = max([len(nameList[i]) for i in range(len(nameList))])
            maxNameLength = max(len("Method"), maxNameLength)
        else:
            nameList = list()
            maxNameLength = len("Method")

        if exact_y_vec is not None:
            maxLength = maxNameLength + 78
        else:
            maxLength = maxNameLength + 25

        funcSource = inspect.getsource(scalarF)
        funcLines = funcSource.split('\n')
        maxTab = len(funcLines[0]) - len(funcLines[0].lstrip())
        for i, line in enumerate(funcLines):
            funcLines[i] = line[maxTab:]
            maxLength = max(maxLength, len(line[maxTab:]) + 10)

        print()
        print(f"{'Result of Approximation(System)':-^{maxLength}}")
        print("Input parameter:")
        for i, line in enumerate(funcLines):
            if i == 0:
                print("function: " + line)
            else:
                print(" " * 10 + line)
        print(f"{'nth x':>8}: {xn}")
        print(f"{'nth yvec':>8}: {yn_vec}")
        print(f"{'h':>8}: {h}")
        print()

        print("Output:")
        print(f"{'Method':^{maxNameLength}}      {'approximated y':^19}", end='')
        if exact_y_vec is not None:
            print(f"    {'error':^19}    {'real y':^24}")
        else:
            print()
        print('-' * maxLength)

        for i in range(len(method_arr)):
            for j in range(len(y_next_arr_vec[0])):
                if j == 0:
                    print(f"{nameList[i]:^{maxNameLength}} :", end='')
                else:
                    print(" " * (maxNameLength + 2), end='')

                print(f"    y{j}: {y_next_arr_vec[i][j]:+12.12f}", end='')
                if exact_y_vec is not None:
                    print(f"     e{j}: {error_arr_vec[i][j]:+12.12f}", end='')
                    if i == 0:
                        print(f"     real y{j}: {exact_y_vec[j]:+12.12f}")
                    else:
                        st = "''"
                        print(f'     {st:^24}')
                else:
                    print()
            print()

        print(f"{'Print finished':-^{maxLength}}")
        print()

    @staticmethod
    def printResultOfIteration(coeff_mat, b_vec, epoch=100):
        from iteration.iteration import Iteration
        x_GS_vec = Iteration.GaussSeidelMethod(coeff_mat, b_vec, epoch=epoch)
        x_J_vec = Iteration.JacobiMethod(coeff_mat, b_vec, epoch=epoch)

        print()
        print(f"{'Result of Iteration':-^{50}}")
        print(f"x from Gauss-Seidel: {x_GS_vec}")
        print(f"{'x from Jacobi':>19}: {x_J_vec}")
        print()
        print("Check Result:")
        print(f"{'vector b':>16}: {b_vec}")
        print(f"matmul of A, xGS: {np.matmul(coeff_mat, x_GS_vec)}")
        print(f"matmul of A, xJ : {np.matmul(coeff_mat, x_J_vec)}")
        print()
        print(f"{'Print finished':-^{50}}")
        print()
