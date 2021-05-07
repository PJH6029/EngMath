import numpy as np

class PDE:
    @staticmethod
    def PoissonEquation(xrange, yrange, h,
                        uL=None, uR=None, uB=None, uT=None, f=None,
                        un_axis=None, un=None,  # for mixed boundary condition
                        boundary_type='Dirichlet'):
        if not f:  # Laplace Equation
            def zero_func(x, y):
                return 0
            poisson_func = zero_func
        else:
            poisson_func = f

        if boundary_type == 'Dirichlet':
            if uL == None or uR == None or uB == None or uT == None :  # all 쓰고싶은데 경계가 0인거 때문에 안됨 ㅜ
                raise Exception("insufficient boundary conditions: uL, uR, uB, uT")
            return PDE.__PoissonEquationWithDirichlet(xrange, yrange, h, uL, uR, uB, uT, poisson_func)
        elif boundary_type.lower() == 'neumann' or 'mixed':
            if not all([un_axis, un]):
                raise Exception("insufficient boundary conditions: un_axis, un")
            if un_axis == 'x' : un_axis = 0
            elif un_axis == 'y' : un_axis = 1
            return PDE.__PoissonEquationWithNeumann(xrange, yrange, h, un_axis, un, poisson_func, uL, uR, uB, uT)

    @staticmethod
    def __PoissonEquationWithDirichlet(xrange, yrange, h, uL, uR, uB, uT, poisson_func):
        from Iteration import Iteration
        x_linspace = np.arange(xrange[0], xrange[1] + h, h)
        y_linspace = np.arange(yrange[0], yrange[1] + h, h)

        m = len(x_linspace)
        n = len(y_linspace)
        vec_len = (m-2) * (n-2)

        if type(uL) == int or float:  # uL is a constant
            left_boundary_values = np.array([uL for _ in range(n-2)])
        else:  # uL is a function
            left_boundary_values = np.array([uL(x_linspace[0], y_linspace[j]) for j in range(1, n-1)])

        if type(uR) == int or type(uR) == float:  # uR is a constant
            right_boundary_values = np.array([uR for _ in range(n-2)])
        else:  # uR is a function
            right_boundary_values = np.array([uR(x_linspace[-1], y_linspace[j]) for j in range(1, n-1)])

        if type(uB) == int or type(uB) == float:  # uB is a constant
            bottom_boundary_values = np.array([uB for _ in range(m-2)])
        else:  # uB is a function
            bottom_boundary_values = np.array([uB(x_linspace[i], y_linspace[0]) for i in range(1, m-1)])

        if type(uT) == int or type(uT) == float:  # uT is a constant
            top_boundary_values = np.array([uT for _ in range(m - 2)])
        else:  # uT is a function
            top_boundary_values = np.array([uT(x_linspace[i], y_linspace[-1]) for i in range(1, m-1)])

        x_grid, y_grid = np.meshgrid(x_linspace[1:-1], y_linspace[1:-1])
        # m-2 x n-2

        x_coordinates = x_grid.ravel()
        y_coordinates = y_grid.ravel()

        # 경계보다 한칸 안에 있는 점들 indices
        left_inner_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[1]))
        right_inner_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[m - 2]))
        bottom_inner_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[1]))
        top_inner_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[n - 2]))

        # f(x,y) 넣어줌
        b_vec = np.array([pow(h, 2) * poisson_func(x_coordinates[i], y_coordinates[i]) for i in range(vec_len)])

        # boundary conditions 넣어줌
        for j, left_index in enumerate(left_inner_boundary_indices):
            b_vec[left_index] -= left_boundary_values[j]
        for j, right_index in enumerate(right_inner_boundary_indices):
            b_vec[right_index] -= right_boundary_values[j]
        for i, bottom_index in enumerate(bottom_inner_boundary_indices):
            b_vec[bottom_index] -= bottom_boundary_values[i]
        for i, top_index in enumerate(top_inner_boundary_indices):
            b_vec[top_index] -= top_boundary_values[i]

        D = -4 * np.identity(m-2) + np.eye(m-2, k=-1) + np.eye(m-2, k=1)
        I = np.identity(m-2)

        A = np.block([[D if i == j else I for i in range(n-2)] for j in range(n-2)])
        # A * u_vec = b_vec

        u_vec = Iteration.GaussSeidelMethod(coeff_mat=A, b_vec=b_vec)

        return u_vec

    @staticmethod
    def __PoissonEquationWithNeumann(xrange, yrange, h, un_axis, un, poisson_func, uL=None, uR=None, uB=None, uT=None):
        if un_axis == 0:
            return PDE.__NeumannWithAxisX(xrange, yrange, h, un, poisson_func, uL, uR, uB, uT)
        elif un_axis == 1:
            return PDE.__NeumannWithAxisY(xrange, yrange, h, un, poisson_func, uL, uR, uB, uT)

    @staticmethod
    def __NeumannWithAxisX(xrange, yrange, h, un, poisson_func, uL=None, uR=None, uB=None, uT=None):
        # uR이 없다는 가정하에 작성
        # 모든 u는 function of x, y

        from Iteration import Iteration
        x_linspace = np.arange(xrange[0], xrange[1] + h, h)
        y_linspace = np.arange(yrange[0], yrange[1] + h, h)

        m = len(x_linspace)
        n = len(y_linspace)

        vec_len = (m - 1) * (n - 2)

        if type(uL) == int or float:  # uL is a constant
            left_boundary_values = np.array([uL for _ in range(n - 2)])
        else:  # uL is a function
            left_boundary_values = np.array([uL(x_linspace[0], y_linspace[j]) for j in range(1, n - 1)])

        if type(uB) == int or type(uB) == float:  # uB is a constant
            bottom_boundary_values = np.array([uB for _ in range(m - 1)])
        else:  # uB is a function
            bottom_boundary_values = np.array([uB(x_linspace[i], y_linspace[0]) for i in range(1, m)])

        if type(uT) == int or type(uT) == float:  # uT is a constant
            top_boundary_values = np.array([uT for _ in range(m - 1)])
        else:  # uT is a function
            top_boundary_values = np.array([uT(x_linspace[i], y_linspace[-1]) for i in range(1, m)])

        x_grid, y_grid = np.meshgrid(x_linspace[1:], y_linspace[1:-1])
        # m-2 x n-1

        x_coordinates = x_grid.ravel()
        y_coordinates = y_grid.ravel()

        # 경계보다 한칸 안에 있는 점들 indices
        left_inner_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[1]))
        bottom_inner_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[1]))
        top_inner_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[n - 2]))

        # un 계산할 경계 indices
        right_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[-1]))

        # f(x,y) 넣어줌
        b_vec = np.array([pow(h, 2) * poisson_func(x_coordinates[i], y_coordinates[i]) for i in range(vec_len)])

        # boundary conditions 넣어줌
        for j, left_index in enumerate(left_inner_boundary_indices):
            b_vec[left_index] -= left_boundary_values[j]
        for i, bottom_index in enumerate(bottom_inner_boundary_indices):
            b_vec[bottom_index] -= bottom_boundary_values[i]
        for i, top_index in enumerate(top_inner_boundary_indices):
            b_vec[top_index] -= top_boundary_values[i]

        # 경계에서 근사치 빼줌
        for right_index in right_boundary_indices:
            b_vec[right_index] -= 2 * h * un(x_coordinates[right_index], y_coordinates[right_index])

        D = -4 * np.identity(m - 1) + np.eye(m - 1, k=-1) + np.eye(m - 1, k=1)
        D[-1][-2] += 1
        I = np.identity(m - 1)

        A = np.block([[D if i == j else I for i in range(n - 2)] for j in range(n - 2)])
        # A * u_vec = b_vec

        u_vec = Iteration.GaussSeidelMethod(coeff_mat=A, b_vec=b_vec)

        return u_vec

    @staticmethod
    def __NeumannWithAxisY(xrange, yrange, h, un, poisson_func, uL=None, uR=None, uB=None, uT=None):
        # uT가 없다는 가정하에 작성
        # 모든 u는 function of x, y

        from Iteration import Iteration
        x_linspace = np.arange(xrange[0], xrange[1] + h, h)
        y_linspace = np.arange(yrange[0], yrange[1] + h, h)

        m = len(x_linspace)
        n = len(y_linspace)

        vec_len = (m-2) * (n-1)

        if type(uL) == int or float:  # uL is a constant
            left_boundary_values = np.array([uL for _ in range(n-1)])
        else:  # uL is a function
            left_boundary_values = np.array([uL(x_linspace[0], y_linspace[j]) for j in range(1, n)])

        if type(uR) == int or type(uR) == float:  # uR is a constant
            right_boundary_values = np.array([uR for _ in range(n-1)])
        else:  # uR is a function
            right_boundary_values = np.array([uR(x_linspace[-1], y_linspace[j]) for j in range(1, n)])

        if type(uB) == int or type(uB) == float:  # uB is a constant
            bottom_boundary_values = np.array([uB for _ in range(m-2)])
        else:  # uB is a function
            bottom_boundary_values = np.array([uB(x_linspace[i], y_linspace[0]) for i in range(1, m-1)])

        x_grid, y_grid = np.meshgrid(x_linspace[1:-1], y_linspace[1:])
        # m-2 x n-1

        x_coordinates = x_grid.ravel()
        y_coordinates = y_grid.ravel()

        # 경계보다 한칸 안에 있는 점들 indices
        left_inner_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[1]))
        right_inner_boundary_indices = np.squeeze(np.where(x_coordinates == x_linspace[m - 2]))
        bottom_inner_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[1]))

        # un 계산할 경계 indices
        top_boundary_indices = np.squeeze(np.where(y_coordinates == y_linspace[-1]))

        # f(x,y) 넣어줌
        b_vec = np.array([pow(h, 2) * poisson_func(x_coordinates[i], y_coordinates[i]) for i in range(vec_len)])

        # boundary conditions 넣어줌
        for j, left_index in enumerate(left_inner_boundary_indices):
            b_vec[left_index] -= left_boundary_values[j]
        for j, right_index in enumerate(right_inner_boundary_indices):
            b_vec[right_index] -= right_boundary_values[j]
        for i, bottom_index in enumerate(bottom_inner_boundary_indices):
            b_vec[bottom_index] -= bottom_boundary_values[i]

        # 경계에서 근사치 빼줌
        for top_index in top_boundary_indices:
            b_vec[top_index] -= 2 * h * un(x_coordinates[top_index], y_coordinates[top_index])

        D = -4 * np.identity(m - 2) + np.eye(m - 2, k=-1) + np.eye(m - 2, k=1)
        I = np.identity(m - 2)

        A = np.block([[D if i == j else (2 * I if i == n-3 and j == n-2 else I) for i in range(n-1)] for j in range(n-1)])
        # A * u_vec = b_vec

        u_vec = Iteration.GaussSeidelMethod(coeff_mat=A, b_vec=b_vec)

        return u_vec

    @staticmethod
    def ParabolicPDE(xrange, trange, initial_f, h, r=1, uL=None, uR=None, plot=False):
        # parabolic PDE: ut = uxx
        # crank-nicolson method

        from Iteration import Iteration
        if plot:
            pass

        if (type(uL) != (int or float)) or (type(uR) != (int or float)):
            raise Exception("uL or uR is function: not implemented yet")
        if r != 1:
            raise Exception("r != 1: not implemented yet")

        k = r * pow(h, 2)
        x_linspace = np.arange(xrange[0], xrange[1] + h, h)
        t_linspace = np.arange(trange[0], trange[1] + k, k)

        m = len(x_linspace)
        n = len(t_linspace)

        # TODO diff < e 등으로 처리해야할듯
        '''  np.sin이 정확한 값을 뱉지를 않아서 예외처리 불가능.
        if uL != initial_f(x_linspace[0]) or uR != initial_f(x_linspace[-1]):
            print(uL, uR, initial_f(x_linspace[0]), initial_f(x_linspace[-1]))
            raise Exception("invalid boundary condition")
        '''
        x_boundary = np.array([uL if i == 0 else (uR if i == m-1 else initial_f(x_linspace[i])) for i in range(m)])
        t_boundary = np.array([uL, uR])

        result = list()
        result.append((t_linspace[0], x_boundary))

        for time in t_linspace[1:]:
            b_vec = np.zeros(m - 2)
            next_f = np.zeros(m)  # 다음 t에 나타날 f의 근사
            next_f[0] = t_boundary[0]
            next_f[-1] = t_boundary[-1]

            # x boundary 빼줌
            for i in range(1, m - 1):
                x_boundary_sum = x_boundary[i - 1] + x_boundary[i + 1]
                b_vec[i - 1] -= x_boundary_sum

            # t boundary 빼줌
            for i in range(-1, 1):
                b_vec[i] -= t_boundary[i]

            D = -4 * np.identity(m - 2) + np.eye(m - 2, k=-1) + np.eye(m - 2, k=1)

            inner_next_f = Iteration.GaussSeidelMethod(coeff_mat=D, b_vec=b_vec)

            next_f[1:-1] = inner_next_f

            next_t_and_f = (time, next_f)
            result.append(next_t_and_f)
            x_boundary = next_f

        return result

    @staticmethod
    def HyperbolicPDE(xrange, trange, initial_f, initial_g, h, r=1, uL=None, uR=None, plot=False):
        # Hyperbolic PDE: utt = uxx
        # f: u(x, 0). g: ut(x, 0)
        from Iteration import Iteration
        if plot:
            pass

        if (type(uL) != (int or float)) or (type(uR) != (int or float)):
            raise Exception("uL or uR is function: not implemented yet")
        if r != 1:
            raise Exception("r != 1: not implemented yet")

        k = h  # r = 1일때
        x_linspace = np.arange(xrange[0], xrange[1] + h, h)
        t_linspace = np.arange(trange[0], trange[1] + k, k)

        m = len(x_linspace)
        n = len(t_linspace)

        x_boundary = np.array([uL if i == 0 else uR if i == m-1 else initial_f(x_linspace[i]) for i in range(m)])
        t_boundary = np.array([uL, uR])

        result = list()
        result.append((t_linspace[0], x_boundary))

        for k, time in enumerate(t_linspace[1:]):
            next_f = np.zeros(m)  # 다음 t에 나타날 f의 근사
            next_f[0] = t_boundary[0]
            next_f[-1] = t_boundary[-1]

            if k == 0:  # first
                for i in range(1, m - 1):
                    x_boundary_sum = x_boundary[i - 1] + x_boundary[i + 1]
                    g_term = k * initial_g(x_linspace[i])
                    next_f[i] = 0.5 * x_boundary_sum + g_term
            else:
                for i in range(1, m - 1):
                    x_boundary_sum = x_boundary[i - 1] + x_boundary[i + 1] - x_previous_boundary[i]
                    next_f[i] = x_boundary_sum

            next_t_and_f = (time, next_f)
            result.append(next_t_and_f)
            x_previous_boundary = x_boundary
            x_boundary = next_f

        return result
def f(x):
    return np.sin(np.pi * x)
def g(x):
    return 0
a = PDE.HyperbolicPDE(xrange=(0, 1), trange=(0, 1.0), initial_f=f, initial_g=g,
                            h=0.2, uL=0, uR=0)
print(a)