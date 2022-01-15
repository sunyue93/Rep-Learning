from scipy.io import loadmat
import numpy as np
import scipy
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy.linalg as npl


class optimal_representation_matrix:
    def __init__(self, beta, p, n, sigma_square, epsilon, beta_star):
        self.beta = beta  # beta vector in the non-asymptotic optimal representation part
        self.p = p  # p is dimension of the features
        self.n = n  # Number of training samples
        self.sigma_square = sigma_square  # Noise variance
        self.epsilon = epsilon  # tiny parameter to prevent divergence of the optimization algorithm (actually not necessary but provides early stopping until some tolerance in the optimization algo.)
        self.expected_error = 0
        self.risk_test = 0
        self.risk_test_no_shaping = 0
        self.beta_star = beta_star

    def objective_func(self, zeta):  # Objective func to minimize, zeta refers to theta in the paper
        return self.sigma_square + self.p * ((self.n / self.p) * (
                np.linalg.norm(np.multiply(self.beta, 1 - zeta)) ** 2) + self.sigma_square / self.p * np.linalg.norm(
            zeta) ** 2) / (self.n - np.linalg.norm(zeta) ** 2)

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
        zeta_optimal = res.x
        #         self.expected_error = self.objective_func(res.x)
        self.expected_error = self.sigma_square + self.p * ((self.n / self.p) * (
                np.linalg.norm(np.multiply(self.beta_star, 1 - zeta_optimal)) ** 2) + self.sigma_square / self.p * np.linalg.norm(zeta_optimal) ** 2) / (
                                          self.n - np.linalg.norm(zeta_optimal) ** 2)

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



p,n,nb_iteration,ksi,sigma_square,epsilon,n_test = 100,40,3,1,0.01,0.0000001,2000  # p = feature dimension, n=number of training samples for the new task, ksi=1(it can be randomly chose), sigma_square=noise variance,
# epsilon=tolerance in the optimization,n_test = number of test samples for the new task

Constant,s  = 25, int(2*p/10)   # Constant=big eigenvalues;  s = effective rank
iota = 0.2

B = np.diag(np.concatenate((1 * np.ones(shape=(s,)), # Covariance of task vectors
                            iota * np.ones(shape=(p-s,))), axis=0))
beta_star = np.sqrt(B).dot(np.random.normal(loc=0, scale=1,size=(p,)))   # New task vector

n_truth = n
X = np.random.normal(loc=0, scale=1, size=(n,p))
y = X.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n,))

X_test = np.random.normal(loc=0, scale=1, size=(n_test,p))
y_test = X_test.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_test,))

ns = [100,200,500,1000,10000] #number of samples per task
k = 500 # number of tasks
rs = [int(n+4*i) for i in range(0,int((p-n+4)/4))]  # Subspace dimensions
ensure_level = 1                        # Level of doing same experiments over and over again and taking average


error, error_opt, error_opt_B_sqrt = np.zeros(shape=(len(ns),len(rs))), np.zeros(shape=(len(ns),len(rs))), np.zeros(shape=(len(ns),len(rs)))  # Errors for the 3 cases : 1sr no shaping,2nd shaping with knowledge of \beta^*, 3rd shaping with knowledge of B matrix
error_theoretic, error_B_sqrt_theoretic = np.zeros(shape=(len(ns),len(rs))), np.zeros(shape=(len(ns),len(rs)))  # theoretical values for the error (we will not use them in the paper)


for _ in range(ensure_level):
#     beta_star = np.sqrt(B).dot(np.random.normal(loc=0, scale=1,size=(p,)))   # New task vector
#     X = np.random.normal(loc=0, scale=1, size=(n_truth,p))
#     y = X.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_truth,))
#     X_test = np.random.normal(loc=0, scale=1, size=(n_test,p))
#     y_test = X_test.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_test,))
    for z in range(len(ns)):
        nt = ns[z]   #number of samples per task
        X_meta = np.random.normal(loc=0,scale=1,size=(k, nt, p))   # meta-train data k=numberoftask,p is the dimension
        beta_meta = np.random.normal(loc=0,scale=1,size=(k,p))     # taks vectors
        y_meta =  np.ones(shape=(k,nt))

        for j in range(k):
            beta_meta[j,:] = np.sqrt(B).dot(beta_meta[j,:].T)      # shaping task vectors

        for j in range(k):
            for i in range(nt):
                y_meta[j,i] = X_meta[j,i,:].dot(beta_meta[j,:].T) + np.sqrt(sigma_square)*np.random.normal(loc=0,scale=1)  #label generation


        # MOM
        B_hat = np.zeros(shape=(p,p))
        for j in range(k):
            avg = np.zeros(shape=(p,))
            for i in range(nt):                                    # Method of Moment estimator as described
                avg += y_meta[j,i]*X_meta[j,i,:]/(nt)

            B_hat += np.outer(avg,avg)



        # PSD Projection of B matrix
        B_averaged = B_hat/(k)
        print(nt)
        print(B)
        print(B_averaged)
#         e_values,e_vectors = npl.eig(B_averaged)
#         e_values_new = np.maximum(e_values,0)
#         B_averaged = e_vectors.dot(np.diag(e_values_new).dot(npl.inv(e_vectors)))

        diagonal = np.diag(B_averaged)
        beta_B_sqrt = np.sqrt(diagonal)    # Getting estimated beta vector as described in the paper
        S = np.linalg.svd(B_averaged)[0]
        for i in range(len(rs)):
            r = rs[i]   # Subspace dimension
            X_r = X.dot(S[:, :r])  # Projection of the data onto the subspace

            beta_hat = np.linalg.lstsq(X_r, y, rcond=None)[0]
            error_cur = (np.linalg.norm(X_test.dot(S[:, :r]).dot(beta_hat) - y_test)) ** 2 / n_test
            error_cur_n = (np.linalg.norm(X_test.dot(S[:, :r]).dot(beta_hat) - y_test)) ** 2 / (np.linalg.norm(y_test) ** 2)  # test Error with no shaping
            error[z,i] += error_cur_n
            zeta = ((n / r)) * np.ones(shape=(r,))

            diagonal = np.diag(B_averaged)
            beta_B_sqrt = np.sqrt(diagonal)
            sigma_square_r = sigma_square + np.linalg.norm(beta_star.T.dot(S[:, r:])) ** 2
            error_theoretic_cur = sigma_square_r + r * ((n / r) * (
                np.linalg.norm(np.multiply(beta_star.T.dot(S[:, :r]), 1 - zeta)) ** 2) + sigma_square_r / r * np.linalg.norm(zeta) ** 2) / (
                                                    n - np.linalg.norm(zeta) ** 2)
            error_theoretic_cur_n = error_theoretic_cur / (np.trace(B)+sigma_square)
            error_theoretic[z,i] += error_theoretic_cur_n

            A = optimal_representation_matrix(beta_B_sqrt.T.dot(S[:, :r]), r, n,
                                              sigma_square + np.linalg.norm(beta_star.T.dot(S[:, r:])) ** 2,
                                              epsilon,beta_star.T.dot(S[:, :r]))  # finding optimal shaping matrix
            lambda_mat = A.lambdas()  # optimal shaping matrix
            X_r_opt = X_r.dot(lambda_mat)  # data after shaping
            beta_hat = (np.linalg.pinv(X_r.dot(lambda_mat))).dot(y)
            error_opt_B_sqrt[z,i] += (np.linalg.norm(X_test.dot(S[:, :r]).dot(lambda_mat).dot(beta_hat) - y_test)) ** 2 / (
                np.linalg.norm(y_test) ** 2)  # test error with shaping
            error_B_sqrt_theoretic[z,i] += A.expected_error / (np.trace(B)+sigma_square)

error, error_theoretic, error_opt_B_sqrt, error_B_sqrt_theoretic = error / ensure_level, error_theoretic / ensure_level, error_opt_B_sqrt / ensure_level, error_B_sqrt_theoretic / ensure_level  # Averaging over ensure_level

# Doing same projection and shaping, then calculating the error. But for the perfect covariance knowledge B. So, no use of MoM
perfect_subspace_error = np.zeros(shape=(len(rs),))
perfect_subspace_error_theoric = np.zeros(shape=(len(rs),))
ps_error, ps_error_opt, ps_error_opt_B_sqrt = np.zeros(shape=(len(rs),)), np.zeros(shape=(len(rs),)), np.zeros(shape=(len(rs),))  # Errors for the 3 cases : 1sr no shaping,2nd shaping with knowledge of \beta^*, 3rd shaping with knowledge of B matrix
ps_error_theoretic, ps_error_B_sqrt_theoretic = np.zeros(shape=(len(rs),)), np.zeros(shape=(len(rs),))  # theoretical values for the error (we will not use them in the paper)
for _ in range(ensure_level):
#     beta_star = np.sqrt(B).dot(np.random.normal(loc=0, scale=1,size=(p,)))   # New task vector
#     X = np.random.normal(loc=0, scale=1, size=(n_truth,p))
#     y = X.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_truth,))
#     X_test = np.random.normal(loc=0, scale=1, size=(n_test,p))
#     y_test = X_test.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_test,))
    S = np.linalg.svd(B)[0]
    for i in range(len(rs)):
        r = rs[i]   # Subspace dimension
        X_r = X.dot(S[:, :r])  # Projection of the data onto the subspace

        diagonal = np.diag(B)
        beta_B_sqrt = np.sqrt(diagonal)
        sigma_square_r = sigma_square + np.linalg.norm(beta_B_sqrt.T.dot(S[:, r:])) ** 2
        error_theoretic_cur = sigma_square_r + r * ((n / r) * (
            np.linalg.norm(np.multiply(beta_B_sqrt.T.dot(S[:, :r]), 1 - zeta)) ** 2) + sigma_square_r / r * np.linalg.norm(zeta) ** 2) / (
                                                n - np.linalg.norm(zeta) ** 2)
        error_theoretic_cur_n = error_theoretic_cur / (np.linalg.norm(y_test) ** 2 / n_test)
        ps_error_theoretic[i] += error_theoretic_cur_n

        A = optimal_representation_matrix(beta_B_sqrt.T.dot(S[:, :r]), r, n,
                                          sigma_square + np.linalg.norm(beta_star.T.dot(S[:, r:])) ** 2,
                                          epsilon,beta_star.T.dot(S[:, :r]))  # finding optimal shaping matrix
        lambda_mat = A.lambdas()  # optimal shaping matrix
        X_r_opt = X_r.dot(lambda_mat)  # data after shaping
        beta_hat = (np.linalg.pinv(X_r.dot(lambda_mat))).dot(y)
        ps_error_opt_B_sqrt[i] += (np.linalg.norm(X_test.dot(S[:, :r]).dot(lambda_mat).dot(beta_hat) - y_test)) ** 2 / (
            np.linalg.norm(y_test) ** 2)  # test error with shaping
        ps_error_B_sqrt_theoretic[i] += A.expected_error / (np.trace(B)+sigma_square)


perfect_subspace_error, perfect_subspace_error_theoric = perfect_subspace_error/ensure_level, perfect_subspace_error_theoric/ensure_level
ps_error, ps_error_theoretic, ps_error_opt_B_sqrt, ps_error_B_sqrt_theoretic = ps_error / ensure_level, ps_error_theoretic / ensure_level, ps_error_opt_B_sqrt / ensure_level, ps_error_B_sqrt_theoretic / ensure_level  # Averaging over ensure_level


# PLOTTING FOR THE OVERPARAMETERIZED CASE
plt.plot(rs, error_B_sqrt_theoretic[0,:],color='b',marker='*')
plt.plot(rs, error_B_sqrt_theoretic[1,:],color='g',marker='o')
plt.plot(rs, error_B_sqrt_theoretic[2,:],color='r',marker='d')
plt.plot(rs, error_B_sqrt_theoretic[3,:],color='k',marker='v')
plt.plot(rs, error_B_sqrt_theoretic[4,:],color='b',marker='v')
plt.plot(rs, ps_error_B_sqrt_theoretic,color='y',marker='^')

plt.axis(ymin=0,ymax=2.5)
plt.xlabel('Representation Dimension', fontsize=20)
plt.ylabel('Few-Shot Test Error', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend([r'$n_{spt}=$'+str(ns[0]), r'$n_{spt}=$'+str(ns[1]),r'$n_{spt}=$'+str(ns[2]),r'$n_{spt}=$'+str(ns[3]),'perfect covariance'],loc='lower left')
plt.grid(True)
plt.show()

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
    # S_hat is p*R
    # does this return eigenspace? I'm assuming it's p*R
    U = S_hat
    Residual = Canonical - U.dot(U.T).dot(Canonical) -  Canonical.dot(U).dot(U.T) + U.dot(U.T).dot(Canonical).dot(U).dot(U.T)
    gamma = p/n
    theta = R/p
    exp_term1 = np.trace(Residual)
    err = (exp_term1 + sigma_square) / (1- theta * gamma)
    return err


##UNDERPARAMETERIZED REGIME FOR ESTIMATED COVARIANCES
# Here we are doing same thing but for the underparameterized case. Namely r<n_fs. So, when we are solving the problem we are doing classical linear regression
# ensure_level = 3
rs1 = [int(4 * i) for i in range(1, int(n / 4))]
all_errors1 = np.zeros(shape=(len(ns), len(rs1)))
error_left_side_theoric = np.zeros(shape=(len(ns), len(rs1)))
error_left_side_experimental = np.zeros(shape=(len(ns), len(rs1)))
for _ in range(ensure_level):
    beta_star = np.sqrt(B).dot(np.random.normal(loc=0, scale=1, size=(p,)))  # New task vector
    X = np.random.normal(loc=0, scale=1, size=(n_truth, p))
    y = X.dot(beta_star) + np.random.normal(loc=0, scale=np.sqrt(sigma_square), size=(n_truth,))
    X_test = np.random.normal(loc=0, scale=1, size=(n_test, p))
    y_test = X_test.dot(beta_star) + np.random.normal(loc=0, scale=np.sqrt(sigma_square), size=(n_test,))
    for z in range(len(ns)):
        nt = ns[z]
        X_meta = np.random.normal(loc=0, scale=1, size=(k, nt, p))
        beta_meta = np.random.normal(loc=0, scale=1, size=(k, p))
        y_meta = np.ones(shape=(k, nt))

        for j in range(k):
            beta_meta[j, :] = np.sqrt(B).dot(beta_meta[j, :].T)

        for j in range(k):
            for i in range(nt):
                y_meta[j, i] = X_meta[j, i, :].dot(beta_meta[j, :].T) + np.sqrt(sigma_square) * np.random.normal(loc=0, scale=1)

        B_hat = np.zeros(shape=(p, p))
        #     An option for MoM
        for j in range(k):
            avg = np.zeros(shape=(p,))
            for i in range(nt):
                avg += y_meta[j, i] * X_meta[j, i, :] / (nt)

            B_hat += np.outer(avg, avg)

        B_averaged = B_hat / (k)
        #         e_values,e_vectors = npl.eig(B_averaged)
        #         e_values_new = np.maximum(e_values,0)
        #         B_averaged = e_vectors.dot(np.diag(e_values_new).dot(npl.inv(e_vectors)))

        diagonal = np.diag(B_averaged)
        beta_B_sqrt = np.sqrt(diagonal)
        B_averaged = np.real(B_averaged)

        S_hat = np.linalg.svd(B_averaged)[0]
        for i in range(len(rs1)):
            r = rs1[i]
            X_r = X.dot(S_hat[:, :r])

            err_left_test_cur = computeLeftError2(np.ones((p,)), np.diag(B), S_hat[:, :r], sigma_square, p, n, r)
            err_left_test_cur /= (np.trace(B) + sigma_square)
            error_left_side_theoric[z, i] += err_left_test_cur

            reg1 = LinearRegression().fit(X_r, y)
            error_left_side_experimental[z,i] += (np.linalg.norm(y_test - reg1.predict(X_test.dot(S_hat[:, :r])))) ** 2 / (
            np.linalg.norm(y_test) ** 2)  # test error for underparameterized case


error_left_side_experimental /= ensure_level
error_left_side_theoric /= ensure_level

##UNDERPARAMETERIZED REGIME for PERFECT COVARİANCE
# Here we are doing same thing but for the underparameterized case. Namely r<n_fs. So, when we are solving the problem we are doing classical linear regression
# ensure_level = 5
rs1 = [int(4*i) for i in range(1,int(n/4))]
all_errors1 = np.zeros(shape=(len(ns), len(rs1)))
error_left_side_theoric_perfect_subpsace = np.zeros(shape=(len(rs1),))
error_left_side_experimental_perfect_subpsace = np.zeros(shape=(len(rs1),))
for _ in range(ensure_level):
    beta_star = np.sqrt(B).dot(np.random.normal(loc=0, scale=1,size=(p,)))   # New task vector
    X = np.random.normal(loc=0, scale=1, size=(n_truth,p))
    y = X.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_truth,))
    X_test = np.random.normal(loc=0, scale=1, size=(n_test,p))
    y_test = X_test.dot(beta_star) + np.random.normal(loc=0,scale=np.sqrt(sigma_square),size=(n_test,))
    S_hat = np.linalg.svd(B)[0]
    for i in range(len(rs1)):
        r = rs1[i]
        X_r = X.dot(S_hat[:,:r])
        err_left_test_cur = computeLeftError2(np.ones((p,)),  np.diag(B), S_hat[:,:r], sigma_square, p, n, r)
        err_left_test_cur /= (np.trace(B)+sigma_square)
        error_left_side_theoric_perfect_subpsace[i] += err_left_test_cur
        reg1 = LinearRegression().fit(X_r, y)
        error_left_side_experimental_perfect_subpsace[i] += (np.linalg.norm(y_test - reg1.predict(X_test.dot(S_hat[:, :r])))) ** 2 / (
        np.linalg.norm(y_test) ** 2)  # test error for underparameterized case

all_errors1 = all_errors1/ensure_level
error_left_side_theoric_perfect_subpsace /= ensure_level
error_left_side_experimental_perfect_subpsace /= ensure_level


# PLOTTING FOR THE UNDERPARAMETERIZED CASE
plt.plot(rs1, error_left_side_theoric[0,:],color='b',marker='*')
plt.plot(rs1, error_left_side_theoric[1,:],color='g',marker='o')
plt.plot(rs1, error_left_side_theoric[2,:],color='r',marker='d')
plt.plot(rs1, error_left_side_theoric[3,:],color='k',marker='v')
plt.plot(rs1, error_left_side_theoric_perfect_subpsace,color='y',marker='^')
print(error_left_side_theoric_perfect_subpsace)
plt.axis(ymin=0,ymax=2)
plt.xlabel('Representation Dimension', fontsize=20)
plt.ylabel('Few-Shot Test Error', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()


plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric[0,:],error_B_sqrt_theoretic[0,:]),axis=0),color='b',marker='*')
plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric[1,:],error_B_sqrt_theoretic[1,:]),axis=0),color='g',marker='o')
plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric[2,:],error_B_sqrt_theoretic[2,:]),axis=0),color='r',marker='d')
plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric[3,:],error_B_sqrt_theoretic[3,:]),axis=0),color='k',marker='v')
plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric[4,:],error_B_sqrt_theoretic[4,:]),axis=0),color='m',marker='v')
plt.plot(np.concatenate((rs1,rs),axis=0),np.concatenate((error_left_side_theoric_perfect_subpsace, ps_error_B_sqrt_theoretic),axis=0),color='y',marker='^')
plt.axis(ymin=0,ymax=2)
plt.xlabel('Representation Dimension', fontsize=20)
plt.ylabel('Few-Shot Test Error', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.legend([r'$n_{spt}=$'+str(ns[0]), r'$n_{spt}=$'+str(ns[1]),r'$n_{spt}=$'+str(ns[2]),r'$n_{spt}=$'+str(ns[3]),r'$n_{spt}=$'+str(ns[4]),'perfect covariance'],loc='lower left')
plt.grid(True)
plt.show()


x1 = np.concatenate((rs1,rs),axis=0)
x2,x3,x4,x5,x6 = x1,x1,x1,x1,x1
y1 = np.concatenate((error_left_side_theoric[0,:],error_B_sqrt_theoretic[0,:]),axis=0)
y2 = np.concatenate((error_left_side_theoric[1,:],error_B_sqrt_theoretic[1,:]),axis=0)
y3 = np.concatenate((error_left_side_theoric[2,:],error_B_sqrt_theoretic[2,:]),axis=0)
y4 = np.concatenate((error_left_side_theoric[3,:],error_B_sqrt_theoretic[3,:]),axis=0)
y5 = np.concatenate((error_left_side_theoric[4,:],error_B_sqrt_theoretic[4,:]),axis=0)
y6 = np.concatenate((error_left_side_theoric_perfect_subpsace, ps_error_B_sqrt_theoretic),axis=0)
plt.plot(x1,y1,color='b',linewidth=3)
plt.plot(x2,y2,color='g',linewidth=3)
#plt.plot(x3,y3,color='y')
plt.plot(x4,y4,color='k',linewidth=3)
#plt.plot(x5,y5,color='m')
plt.plot(x6,y6,color='r',linewidth=3)
plt.axis(xmin=40,ymin=0,ymax=2)
plt.xlabel('Representation Dimension', fontsize=20)
plt.ylabel('Few-Shot Test Error', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend([r'$n_1=$'+str(ns[0]),r'$n_1=$'+str(ns[1]),r'$n_1=$'+str(ns[3]),'perfect covariance'],loc='upper right',fontsize=12)
plt.tight_layout()
plt.grid(True)

