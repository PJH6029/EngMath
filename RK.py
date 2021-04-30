import numpy as np
from Utils import Utils
np.set_printoptions(precision=12)

class RK:
    @staticmethod
    def get_k_vector(f, xn, yn, h):
        k_vector = np.array([0 for _ in range(6)], dtype=np.float64)  # k1 ~ k6
        h_coeff = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
        k_coeff = np.array([[1 / 4, 0, 0, 0, 0],
                            [3 / 32, 9 / 32, 0, 0, 0],
                            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
                            [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
                            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
        k_vector[0] = h * f(xn, yn)
        for i in range(1, 6):
            k_vector[i] = h * f(xn + h_coeff[i] * h,
                                yn + np.matmul(k_coeff, k_vector[:-1].transpose())[i-1])
        return k_vector

    @staticmethod
    def __RKFifth_with_kvector(yn, kvec):
        gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        y_next = yn + np.dot(gamma, kvec)
        return y_next

    @staticmethod
    def __RKFourth_with_kvector(yn, kvec):
        gamma = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5])
        y_next = yn + np.dot(gamma, kvec)
        return y_next

    @staticmethod
    def get_k_mat(scalarF, xn, yn_vec, h):
        F_vec = Utils.get_F_from_ODE(scalarF)
        k_vec0 = F_vec(xn, *yn_vec)  # for getting len of vec
        k_mat = np.array([[0 for _ in range(len(k_vec0))] for __ in range(6)], dtype=np.float64) # kvec1 ~ kvec6
        h_coeff = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
        k_coeff = np.array([[1 / 4, 0, 0, 0, 0],
                            [3 / 32, 9 / 32, 0, 0, 0],
                            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
                            [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
                            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
        k_mat[0, :] = h * k_vec0
        for i in range(1, 6):
            k_mat[i] = h * F_vec(xn + h_coeff[i] * h,
                                 *(yn_vec + np.matmul(k_coeff, k_mat[:-1])[i-1]))
        return k_mat  # 6 x n

    @staticmethod
    def __RKFifth_with_kmat(yn_vec, kmat):  # kmat: 6xn
        gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        y_next_vec = yn_vec + np.matmul(kmat.transpose(), gamma)
        return y_next_vec

    @staticmethod
    def __RKFourth_with_kmat(yn_vec, kmat):  # kmat: 5xn
        gamma = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5])
        y_next_vec = yn_vec + np.matmul(kmat.transpose(), gamma)
        return y_next_vec

    @staticmethod
    def RKFifth(f, xn, yn, h):
        k_vector = RK.get_k_vector(f, xn, yn, h)
        return RK.__RKFifth_with_kvector(yn, k_vector)

    @staticmethod
    def RKFourth(f, xn, yn, h):
        k_vector = RK.get_k_vector(f, xn, yn, h)
        return RK.__RKFourth_with_kvector(yn, k_vector[:5])

    @staticmethod
    def RKClassic(f, xn, yn, h):
        k1 = h * f(xn, yn)
        k2 = h * f(xn + 0.5 * h, yn + 0.5 * k1)
        k3 = h * f(xn + 0.5 * h, yn + 0.5 * k2)
        k4 = h * f(xn + h, yn + k3)

        y_next = yn + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next

    @staticmethod
    def RKClassicForSystem(scalarF, xn, yn_vec, h): # F: f_scalar
        F_vec = Utils.get_F_from_ODE(scalarF)
        k1_vec = h * F_vec(xn, *yn_vec)
        k2_vec = h * F_vec(xn + 0.5 * h, *(yn_vec + 0.5 * k1_vec))
        k3_vec = h * F_vec(xn + 0.5 * h, *(yn_vec + 0.5 * k2_vec))
        k4_vec = h * F_vec(xn + h, *(yn_vec + k3_vec))

        y_next_vec = yn_vec + (1/6) * (k1_vec + 2 * k2_vec + 2 * k3_vec + k4_vec)
        return y_next_vec

    # Fifth, Fourth RK for system은 교재에는 x
    @staticmethod
    def RKFifthForSystem(scalarF, xn, yn_vec, h):
        k_mat = RK.get_k_mat(scalarF, xn, yn_vec, h)
        return RK.__RKFifth_with_kmat(yn_vec, k_mat)

    @staticmethod
    def RKFourthForSystem(scalarF, xn, yn_vec, h):
        k_mat = RK.get_k_mat(scalarF, xn, yn_vec, h)
        return RK.__RKFourth_with_kmat(yn_vec, k_mat[:5])