from scipy.io import loadmat
import numpy as np
import scipy
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def computeLeftError(bSi, B, sigma_square, p, n, R):
    # bsi and B are vectors
    gamma = p/n
    theta = R/p
    idx_trun = int(p * theta)
    trun_covs = B * bSi
    exp_term_1 = np.sum(trun_covs[idx_trun:])
    err = (exp_term_1 + sigma_square) / (1- theta * gamma)
    return err

def computeLeftError2(bSi, B, S_hat, sigma_square, p, n, R):
    # bsi and B are vectors
    # B is the true B, not send B_average here
    # make it back to matrix
    Canonical = np.diag(bSi*B)
    U = S_hat
    Residual = Canonical - U.dot(U.T).dot(Canonical) -  Canonical.dot(U).dot(U.T) + U.dot(U.T).dot(Canonical).dot(U).dot(U.T)
    gamma = p/n
    theta = R/p
    exp_term1 = np.trace(Residual)
    err = (exp_term1 + sigma_square) / (1- theta * gamma)
    return err

class optimal_representation_matrix:
    def __init__(self, beta, p, n, sigma_square, epsilon):
        self.beta = beta  # beta vector in the non-asymptotic optimal representation part
        self.p = p  # p is dimension of the features
        self.n = n  # Number of training samples
        self.sigma_square = sigma_square  # Noise variance
        self.epsilon = epsilon  # tiny parameter to prevent divergence of the optimization algorithm (actually not necessary but provides early stopping until some tolerance in the optimization algo.)
        self.expected_error = 0
        self.risk_test = 0
        self.risk_test_no_shaping = 0

    def objective_func(self, zeta):  # Objective func to minimize, zeta refers to theta in the paper
        return self.sigma_square + self.p * ((self.n / self.p) * (
            np.linalg.norm(np.multiply(self.beta, 1 - zeta)) ** 2) + self.sigma_square / self.p * np.linalg.norm(
            zeta) ** 2) / (
                                       n - np.linalg.norm(zeta) ** 2)

    '''
    Compute ksi by binary search
    '''

    def func_ksi(self, B, ksi):
        ksi_B = ksi * B
        ksi_B_frac = ksi_B / (1 + ksi_B)
        return np.sum(ksi_B_frac)

    def ksi_solver(self, n, p, B, lower_bd, upper_bd, ksi_iters):
        left = lower_bd
        right = upper_bd
        while self.func_ksi(B, right) < n:
            left = right
            right *= 2
        T = ksi_iters
        for i in range(T):
            mid = (left + right) / 2
            n_mid = self.func_ksi(B, mid)
            if n_mid > n:
                right = mid
            else:
                left = mid
        return mid

    '''
    Above: Compute ksi by binary search
    '''

    def lambdas(self):
        zeta = np.random.uniform(0, 1, size=(self.p,))
        cons = ({'type': 'eq',
                 'fun': lambda x: np.sum(x) - self.n})
        res = minimize(self.objective_func, zeta, constraints=cons,
                       bounds=[(0, 1 - self.epsilon) for _ in range(self.p)],
                       # Optimization algorithm over theta in the paper (we call it zeta here)
                       method='trust-constr', options={'disp': True, 'maxiter': 3000})
        self.expected_error = self.objective_func(res.x)
        lambda_vector = np.zeros(shape=(self.p,))
        for i in range(self.p):
            ksi = 1
            # lambda_vector[i] = res.x[i] / (ksi * (1 - res.x[i]))
            lambda_vector[i] = res.x[i] / (ksi * (1 - res.x[i]))
        lambda_sqrt_vector = np.sqrt(lambda_vector)
        lambda_matrix = np.diag(lambda_sqrt_vector)

        '''
        Check risk 
        '''
        ksi = self.ksi_solver(self.n, self.p, lambda_vector, 0, 10, 100)
        gamma = (sigma_square / self.p + np.mean(self.beta ** 2 * lambda_vector / (1 + ksi * lambda_vector) ** 2)) \
                / (np.mean(1 / (1 + 1 / (ksi * lambda_vector)) ** 2) * self.p / self.n)
        risk_test_1 = np.mean(lambda_vector * self.beta ** 2 / (1 + ksi * lambda_vector) ** 2)
        risk_test_2 = self.p / self.n * gamma * np.mean((1 + 1 / (lambda_vector)) ** (-2))
        risk_test = risk_test_1 + risk_test_2
        risk_test *= self.p
        risk_test += sigma_square
        self.risk_test = risk_test

        ## No-Shaping Theoretical
        lambda_vector = np.ones(shape=(self.p,))
        ksi = self.ksi_solver(self.n, self.p, lambda_vector, 0, 10, 100)
        gamma = (sigma_square / self.p + np.mean(self.beta ** 2 * lambda_vector / (1 + ksi * lambda_vector) ** 2)) \
                / (np.mean(1 / (1 + 1 / (ksi * lambda_vector)) ** 2) * self.p / self.n)
        risk_test_1 = np.mean(lambda_vector * self.beta ** 2 / (1 + ksi * lambda_vector) ** 2)
        risk_test_2 = self.p / self.n * gamma * np.mean((1 + 1 / (lambda_vector)) ** (-2))
        risk_test = risk_test_1 + risk_test_2
        risk_test *= self.p
        risk_test += sigma_square
        self.risk_test_no_shaping = risk_test
        ## End of No-Shaping Theoretical


        return lambda_matrix  # Returns the optimal shaping matrix


K = 1
p, n, nb_iteration, ksi, sigma_square, epsilon, n_test = 100, 40, 3, 1, 0.05, 0.0001, 2000  # p = feature dimension, n=number of training samples for the new task, ksi=1(it can be randomly chose), sigma_square=noise variance,
# epsilon=tolerance in the optimization,n_test = number of test samples for the new task

Constant, s = 25, int(2 * p / 10)  # Constant=big eigenvalues;  s = effective rank
iota = 0.1
B = np.diag(np.concatenate((1 * np.ones(shape=(s,)),  # Covariance of task vectors
                            iota * np.ones(shape=(p-s,))), axis=0))

rs = [int(n + 4 * i) for i in
      range(int((p - n + 4) / 4))]  # Subspace representation dimension for overparameterized case
error, error_opt, error_opt_B_sqrt = np.zeros(shape=(len(rs),)), np.zeros(shape=(len(rs),)), np.zeros(shape=(len(
    rs),))  # Errors for the 3 cases : 1sr no shaping,2nd shaping with knowledge of \beta^*, 3rd shaping with knowledge of B matrix
error_theoretic, error_B_sqrt_theoretic = np.zeros(shape=(len(rs),)), np.zeros(
    shape=(len(rs),))  # theoretical values for the error (we will not use them in the paper)
ensure_level = 5

for _ in range(ensure_level):
    beta = np.sqrt(B).dot(np.random.normal(loc=0, scale=1, size=(p,)))  # task vector generation

    X = np.random.normal(loc=0, scale=1, size=(n, p))  # Data samples generation
    y = X.dot(beta) + np.random.normal(loc=0, scale=np.sqrt(sigma_square), size=(n,))  # Label generation

    X_test = np.random.normal(loc=0, scale=1, size=(n_test, p))  # Data samples generation
    y_test = X_test.dot(beta) + np.random.normal(loc=0, scale=np.sqrt(sigma_square), size=(n_test,))  # Label generation

    diagonal = np.diag(B)
    beta_B_sqrt = np.sqrt(
        diagonal)  # we will do the shaping as if the task vector is square roots of the diagonal values of the B matrix

    S = np.linalg.svd(B)[0]

    # beta = np.sqrt(B)
    # for i in range(len(rs) - 1, len(rs)):
    for i in range(len(rs)):
        r = rs[len(rs) - 1 - i]  # Subspace dimension
        X_r = X.dot(S[:, :r])  # Subspace projection of the data
        diagonal = np.diag(B)
        beta_B_sqrt = np.sqrt(diagonal)

        # beta_hat = np.linalg.pinv(X_r).dot(y)
        beta_hat = np.linalg.lstsq(X_r, y, rcond=None)[0]
        error_cur = (np.linalg.norm(X_test.dot(S[:, :r]).dot(beta_hat) - y_test)) ** 2 / n_test
        error_cur_n = (np.linalg.norm(X_test.dot(S[:, :r]).dot(beta_hat) - y_test)) ** 2 / (
            np.linalg.norm(y_test) ** 2)  # test Error with no shaping
        error[i] += error_cur_n


        zeta = ((n / r)) * np.ones(shape=(r,))
        diagonal = np.diag(B)
        beta_B_sqrt = np.sqrt(diagonal)
        sigma_square_r = sigma_square + np.linalg.norm(beta_B_sqrt.T.dot(S[:, r:])) ** 2
        error_theoretic_cur = sigma_square_r + r * ((n / r) * (
            np.linalg.norm(np.multiply(beta_B_sqrt.T.dot(S[:, :r]), 1 - zeta)) ** 2) + sigma_square_r / r * np.linalg.norm(zeta) ** 2) / (
                                                 n - np.linalg.norm(zeta) ** 2)
        error_theoretic_cur_n = error_theoretic_cur / (np.linalg.norm(y_test) ** 2 / n_test)
        error_theoretic[i] += error_theoretic_cur_n

        ##error with shaping
        #diagonal = np.diag(B)
        #beta_B_sqrt = np.sqrt(diagonal)
        A = optimal_representation_matrix(beta_B_sqrt.T.dot(S[:, :r]), r, n,
                                          sigma_square + np.linalg.norm(beta_B_sqrt.T.dot(S[:, r:])) ** 2,
                                          epsilon)  # finding optimal shaping matrix
        lambda_mat = A.lambdas()  # optimal shaping matrix
        # lambda_mat = lambda_mat.dot(lambda_mat).dot(lambda_mat).dot(lambda_mat).dot(lambda_mat).dot(lambda_mat)
        # error_theoretic[i] += A.risk_test_no_shaping / (np.linalg.norm(y_test) ** 2 / n_test)
        X_r_opt = X_r.dot(lambda_mat)  # data after shaping

        beta_hat = (np.linalg.pinv(X_r.dot(lambda_mat))).dot(y)


        error_opt_B_sqrt[i] += (np.linalg.norm(X_test.dot(S[:, :r]).dot(lambda_mat).dot(beta_hat) - y_test)) ** 2 / (
            np.linalg.norm(y_test) ** 2)  # test error with shaping

        error_B_sqrt_theoretic[i] += A.expected_error / (np.linalg.norm(y_test) ** 2 / n_test)

error, error_theoretic, error_opt_B_sqrt, error_B_sqrt_theoretic = error / ensure_level, error_theoretic / ensure_level, error_opt_B_sqrt / ensure_level, error_B_sqrt_theoretic / ensure_level  # Averaging over ensure_level

## UNDERPARAMETERIZED CASE
# We are doing same thing here. Namely calculating error. But note that as we are in underparameterized case, shaping will be meaningless. Also we are solving problem using linear regression solver

rs1 = [int(2 * i) for i in range(2, int(n / 2))]
S = np.linalg.svd(B)[0]
error1 = np.zeros(shape=(len(rs1),))  # error for underparameterized case
error_intuitive_left_side = np.zeros(shape=(len(rs1),))
error_intuitive_left_side_2 = np.zeros(shape=(len(rs1),))
ensure_level = 80
for _ in range(ensure_level):
    # for i in range(len(rs1) - 6, len(rs1)):
    beta = np.sqrt(B).dot(np.random.normal(loc=0, scale=1, size=(p,)))
    for i in range(len(rs1)):
        X = np.random.normal(loc=0, scale=1, size=(n, p))  # data generation
        y = X.dot(beta) + np.random.normal(loc=0, scale=np.sqrt(sigma_square), size=(n,))  # label generation

        X_test = np.random.normal(loc=0, scale=1, size=(n_test, p))  # test data generation
        y_test = X_test.dot(beta) + np.random.normal(loc=0, scale=np.sqrt(sigma_square),
                                                     size=(n_test,))  # test label generation

        r = rs1[i]
        X_r = X.dot(S[:, :r])
        # lambda1 = np.diag(np.random.uniform(low=0.001,high=5,size=(r,)))
        # X_r_1= X_r.dot(lambda1)
        theta_for_left_side = np.concatenate(( np.ones(shape=(r,)), ((n-r)/(p-r)) * np.ones(shape=(p-r,)),  # Covariance of task vectors
                                                                                          ), axis=0)
        error_intuitive_left_side[i] += ((n/p)*np.linalg.norm(beta*(1-theta_for_left_side))**2+sigma_square*np.linalg.norm(theta_for_left_side)**2)/((n-np.linalg.norm(theta_for_left_side)**2))


        # err_left_test_cur = computeLeftError(np.ones((p,)),  np.diag(B), sigma_square, p, n, r)
        err_left_test_cur = computeLeftError2(np.ones((p,)),  np.diag(B), S[:,:r],sigma_square, p, n, r)
        err_left_test_cur /= (np.linalg.norm(y_test) ** 2 / n_test)
        error_intuitive_left_side_2[i] += err_left_test_cur

        reg1 = LinearRegression().fit(X_r, y)
        error1[i] += (np.linalg.norm(y_test - reg1.predict(X_test.dot(S[:, :r])))) ** 2 / (
            np.linalg.norm(y_test) ** 2)  # test error for underparameterized case

error1 = error1 / ensure_level  # averaging
error_intuitive_left_side = error_intuitive_left_side / ensure_level
error_intuitive_left_side_2 = error_intuitive_left_side_2 / ensure_level


# PLIOTTING EXPERİMENTAL ERRORS

plt.figure(figsize=(8, 6))
plt.plot(np.concatenate((rs1,rs),axis=0), np.concatenate((error1,np.flip(error)),axis=0),color='r',marker='o',linewidth=0)
plt.plot(rs, np.flip(error_opt_B_sqrt),color='b',marker='o',linewidth=0)
plt.plot(rs[1:], (np.flip(error_theoretic))[1:],color='r')
plt.plot(rs[1:], (np.flip(error_B_sqrt_theoretic))[1:],color='b')
plt.plot(rs1,error_intuitive_left_side_2,color='r')
plt.axis(ymin=0,ymax=2)
plt.xlabel('Representation Dimension', fontsize=20)
plt.ylabel('Few-Shot Test Error', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['no shaping experimental','optimal shaping experimental','no shaping theoretical','optimal shaping theoretical'], loc='upper right',fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()