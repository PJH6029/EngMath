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

