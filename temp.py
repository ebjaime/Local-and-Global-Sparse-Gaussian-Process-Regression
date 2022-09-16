import pandas as pd
import scipy

import GPflow.gpflow as gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GPflow.gpflow.config import default_float, default_jitter
from GPflow.gpflow.utilities import print_summary, set_trainable, to_default_float
from GPflow.gpflow.models import maximum_log_likelihood_objective, training_loss_closure
from GPflow.gpflow.ci_utils import ci_niter

from sklearn.metrics import mean_squared_error

from timeit import default_timer as timer

import sys

# Generic functions

def plot_univariate(model, color, ax, lims=[-1,11], show_xs=True, show_ind_locations=True):
    x = model.data[0]
    y = model.data[1]
    xx = np.linspace(lims[0], lims[1], 100).reshape(-1, 1)
    mu, var = model.predict_y(xx)
    ax.plot(xx, mu, color, lw=2)
    ax.fill_between(
        xx[:, 0],
        mu[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mu[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color=color,
        alpha=0.2,
    )
    if show_xs:
        ax.plot(x, y, "k,", mew=2)
    if show_ind_locations:
        try:
            ax.plot(np.array(model.inducing_variable.variables[0]),
                    np.repeat(min(mu[:, 0] - 1.96 * np.sqrt(var[:, 0])) - 1, N_ind),
                    'rx')
        except:
            pass
    # Plot title
    ax.set_title(type(model).__name__)
    ax.set_xlim(lims[0], lims[1])


def create_models(data,
                  inducing_variable,
                  optimize=True,
                  opt=gpflow.optimizers.Scipy(),  # BFGS
                  niter=100,
                  verbose=False
                  ):
    ndim=data[0].shape[1]
    kernel = gpflow.kernels.SquaredExponential() # REVIEW: check other kernels
    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(ndim))
    m1 = gpflow.models.GPR(data, kernel=kernel)

    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(ndim))
    m2 = gpflow.models.SGPR(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m2.inducing_variable, False)

    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(ndim))
    m3 = gpflow.models.GPRFITC(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m3.inducing_variable, False)

    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(ndim))
    m4 = gpflow.models.GPRPITC(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m4.inducing_variable, False)

    # kernel = gpflow.kernels.Matern32(lengthscales=np.ones(ndim))
    # m5 = gpflow.models.GPRPIC(
    #     data, kernel=kernel, inducing_variable=inducing_variable
    # )
    # set_trainable(m5.inducing_variable, False)


    models = [m1,m2,m3,m4]

    if verbose:
        for model in models:
            print_summary(model)

    times = []
    if optimize:
        # Optimize models
        for model in models:
            print("Starting training of ", type(model).__name__)
            loss_closure = training_loss_closure(model, data)
            start = timer()
            opt.minimize(
                loss_closure,
                variables=model.trainable_variables,
                options=dict(maxiter=ci_niter(100))
            )
            end = timer()
            print("Finished training of ", type(model).__name__)
            times.append(end-start)


    return models, times


# m.predict_f -> returns the mean and variance of 𝑓 at the points Xnew
# m.predict_f_samples -> returns samples of the latent function.
# m.predict_y -> returns the mean and variance of a new data point (that is, it includes the noise variance).
def mse_test(model, test_data):
    test_set, test_labels = test_data
    X_t = tf.convert_to_tensor(test_set, dtype=default_float())
    mean, var = model.predict_y(X_t)
    return mean_squared_error(mean, test_labels)


# REVIEW: is this the correct formula?
# m.predict_density -> returns the log density of the observations Ynew at Xnew.
def nlpd_test(model, test_data):
    N = test_data[0].shape[0]
    return 1/N*np.sum(model.predict_log_density(test_data))


def error_vs_time(times, mses, nlpds,
                  markers=['.', '.', 'o', '+'],
                  colors=['black', 'red', 'blue', 'black'],
                  mfcs=[False, False, True, False],
                  model_names=['GPR', 'SGPR', 'FITC', 'PITC']):

    fig, axs = plt.subplots(1, 2, figsize=(16,8))

    # MSE vs time/s
    for i in range(len(times)):
        axs[0].scatter(times[i],
                       mses[i],
                       marker=markers[i],
                       facecolors='none' if mfcs[i] else colors[i],
                       edgecolors=colors[i],
                       label=model_names[i])

    axs[0].legend()
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("time/s")
    axs[0].set_title("MSE vs time")

    # NLPD vs time/s
    for i in range(len(times)):
        axs[1].scatter(times[i],
                       nlpds[i],
                       marker=markers[i],
                       facecolors='none' if mfcs[i] else colors[i],
                       edgecolors=colors[i],
                       label=model_names[i])

    axs[1].legend()
    axs[1].set_ylabel("NLPD")
    axs[1].set_xlabel("time/s")
    axs[1].set_title("NLPD vs time")

# #####################################
# GAUSSIAN PROCESS REGRESSION EXAMPLE (d=2)
# #####################################

# Load data
np.random.seed(42)
N = 1000
N_t = 100
N_ind = 10
X = np.random.rand(N, 1) * 10
Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + np.random.randn(*X.shape) * 0.4
Xtest = np.random.rand(N_t, 1) * 10
# _ = plt.plot(X, Y, "kx", mew=2)
# plt.show()

data = (
    tf.convert_to_tensor(X, dtype=default_float()),
    tf.convert_to_tensor(Y, dtype=default_float()),
)
X_ind = X[0:N-1:int(N/N_ind)]
inducing_variable = tf.convert_to_tensor(X_ind, dtype=default_float())

# Create models
models, _ = create_models(data, inducing_variable, optimize=False)

GPR, SGPR, FITC, PITC = models  # REVIEW: find out what is so slow in PITC (see testing area)
LocalGPR = gpflow.models.LocalGPR(data,gpflow.kernels.SquaredExponential(), num_blocks=10)
LocalGPR.optimize()


# Plot results
f, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=False, sharey=False)
plot_univariate(GPR, "C0", ax[0, 0], show_xs=True)
plot_univariate(SGPR, "C1", ax[0, 1], show_xs=True)
plot_univariate(FITC, "C2", ax[1, 0], show_xs=True)
plot_univariate(PITC, "C3", ax[1, 1], show_xs=True, lims=[-1,6])
# plot_univariate(PIC, "C4", ax[2, 0], show_xs=False)
plot_univariate(LocalGPR, "C5", ax[2, 1], show_xs=True)
plt.show()
sys.exit()


# #####################################
# TODO: KIN40K (d=9)
# #####################################
# 40,000 records describing the location of a robotic arm as a function of an 8-dimensional control input

# Load data
kin40k = np.loadtxt("data/kin40k/kin40k_train_data.asc") # Toy example of kin40k
kin40k_l = np.loadtxt("data/kin40k/kin40k_train_labels.asc").reshape(-1, 1) # Toy example of kin40k
kin40k_test = np.loadtxt("data/kin40k/kin40k_test_data.asc", skiprows=29000) # Toy example of kin40k
kin40k_test_l = np.loadtxt("data/kin40k/kin40k_test_labels.asc", skiprows=29000).reshape(-1, 1) # Toy example of kin40k
N = len(kin40k)
N_t = len(kin40k_test)
N_ind = len(kin40k) * 0.01  # REVIEW: Proportion of inducing points?
n_dim = kin40k.shape[1]

kin40k_data = (
    tf.convert_to_tensor(kin40k, dtype=default_float()),
    tf.convert_to_tensor(kin40k_l, dtype=default_float()),
)
kin40k_test_data = (
    tf.convert_to_tensor(kin40k_test, dtype=default_float()),
    tf.convert_to_tensor(kin40k_test_l, dtype=default_float()),
)
kin40k_ind = kin40k[0:N-1:int(N/N_ind)]
kin40k_ind = tf.convert_to_tensor(kin40k_ind, dtype=default_float())

# https://github.com/GPflow/GPflow/issues/1606
models, times = create_models(kin40k_data, kin40k_ind)
# GPR, SGPR, FITC, PITC, PIC = models
SGPR, FITC, PITC = models

LocalGPR = gpflow.models.LocalGPR(kin40k_data,  gpflow.kernels.SquaredExponential(lengthscales=np.ones(n_dim)), num_blocks=100)
times.append(sum(LocalGPR.optimize()))

# Evaluation of kin40k
kin40k_mses = [#mse_test(GPR, kin40k_test_data),
               mse_test(SGPR, kin40k_test_data),
               mse_test(FITC, kin40k_test_data),
               # mse_test(PITC, kin40k_test_data),
               mse_test(LocalGPR, kin40k_test_data)]

kin40k_nlpd = [#nlpd_test(GPR, kin40k_test_data),
               nlpd_test(SGPR, kin40k_test_data),
               nlpd_test(FITC, kin40k_test_data),
               #nlpd_test(PITC, kin40k_test_data)
               nlpd_test(LocalGPR, kin40k_test_data)]


error_vs_time(times, kin40k_mses, kin40k_nlpd)
plt.show()


# #####################################
# TODO: SARCOS
# #####################################
# The data relates to an inverse dynamics problem for a seven degrees-of-freedom SARCOS anthropomorphic robot arm.
# The task is to map from a 21-dimensional input space (7 joint positions, 7 joint velocities, 7 joint accelerations)
# to the corresponding 7 joint torques.
# There are 44,484 training examples and 4,449 test examples. The first 21 columns are the input variables,
# and the 22nd column is used as the target variable.

# Load data
sarcos = scipy.io.loadmat('data/sarcos/sarcos_inv.mat')['sarcos_inv']
sarcos_l = sarcos[:,-1]
sarcos = sarcos[:,:-1]
sarcos_test = scipy.io.loadmat('data/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
sarcos_test_l = sarcos_test[:,-1]
sarcos_test = sarcos_test[:,:-1]

N = len(sarcos)
N_t = len(sarcos_test)
N_ind = len(sarcos) * 0.01  # REVIEW: Proportion of inducing points?
n_dim = sarcos.shape[1]

sarcos_data = (
    tf.convert_to_tensor(sarcos, dtype=default_float()),
    tf.convert_to_tensor(sarcos_l, dtype=default_float()),
)
sarcos_test_data = (
    tf.convert_to_tensor(sarcos_test, dtype=default_float()),
    tf.convert_to_tensor(sarcos_test_l, dtype=default_float()),
)
sarcos_ind = sarcos[0:N-1:int(N/N_ind)]
sarcos_ind = tf.convert_to_tensor(sarcos_ind, dtype=default_float())

# Training of SARCOS
models, times = create_models(sarcos_data, gpflow.kernels.Matern32(lengthscales=np.ones(n_dim)), sarcos_ind)
GPR, SGPR, FITC, PITC = models

# Evaluation of SARCOS
sarcos_mses = [mse_test(GPR, sarcos_test_data),
               mse_test(SGPR, sarcos_test_data),
               mse_test(FITC, sarcos_test_data),
               mse_test(PITC, sarcos_test_data)]

sarcos_nlpd = [nlpd_test(GPR, sarcos_test_data),
               nlpd_test(SGPR, sarcos_test_data),
               nlpd_test(FITC, sarcos_test_data),
               nlpd_test(PITC, sarcos_test_data)]

error_vs_time(times, sarcos_mses, sarcos_nlpd)
plt.show()

# #####################################
# TODO: ABALONE
# #####################################
# Predicting the age of abalone from physical measurements. The age of abalone is determined by
# cutting the shell through the cone, staining it, and counting the number of rings through a
# microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain,
# are used to predict the age.

abalone = pd.read_csv('data/abalone/abalone.data', header=None)
N = len(abalone)
test_pct = .2
abalone_test = abalone.iloc[:int(N*test_pct),:]
abalone_test_l = abalone_test.iloc[:,-1]
abalone_test = abalone_test.iloc[:,:-1].drop(0, axis=1)

abalone = abalone.iloc[int(N*test_pct):,:]
abalone_l = abalone.iloc[:,-1]
abalone = abalone.iloc[:,:-1].drop(0, axis=1)

N = len(abalone)
N_t = len(abalone_test)
N_ind = len(abalone) * 0.01
n_dim = abalone.shape[1]

abalone_data = (
    tf.convert_to_tensor(abalone, dtype=default_float()),
    tf.convert_to_tensor(abalone_l, dtype=default_float()),
)
abalone_test_data = (
    tf.convert_to_tensor(abalone_test, dtype=default_float()),
    tf.convert_to_tensor(abalone_test_l, dtype=default_float()),
)
abalone_ind = abalone[0:N-1:int(N/N_ind)]
abalone_ind = tf.convert_to_tensor(abalone_ind, dtype=default_float())

# Training of SARCOS
models, times = create_models(abalone_data, gpflow.kernels.Matern32(lengthscales=np.ones(n_dim)), abalone_ind)
GPR, SGPR, FITC, PITC = models

# Evaluation of SARCOS
abalone_mses = [mse_test(GPR, abalone_test_data),
               mse_test(SGPR, abalone_test_data),
               mse_test(FITC, abalone_test_data),
               mse_test(PITC, abalone_test_data)]

abalone_nlpd = [nlpd_test(GPR, abalone_test_data),
               nlpd_test(SGPR, abalone_test_data),
               nlpd_test(FITC, abalone_test_data),
               nlpd_test(PITC, abalone_test_data)]

error_vs_time(times, abalone_mses, abalone_nlpd)
plt.show()


# #####################################
# EXTERNAL DATASET
# #####################################

# TODO: external dataset


# #####################################
# TESTING PLAYGROUND
# #####################################
# from GPflow.gpflow.covariances.dispatch import Kuf, Kuu
# # m3=FITC
# m4=PITC
# # m5=PIC
# # #######################################
# # # Test PI(T)C "commont_terms()" method
# X_data, Y_data =m4.data
# N = X_data.shape[0]
# num_inducing = m4.inducing_variable.num_inducing
# err = Y_data - m4.mean_function(X_data)  # size [N, R] = [1000, 1]
#
# m4.construct_blocks()
#
# # Resort X_data by K-means clustering
# original_index = np.array(range(N))
# sort_by_block = np.argsort(m4.blocks)
# X_data = tf.gather(X_data, sort_by_block) # [x for _, x in np.sort(zip(self.blocks, X_data))]
# blocks = m4.blocks[sort_by_block]
# original_index = original_index[sort_by_block]
#
#
# full_cov = m4.kernel(X_data, full_cov=True)
# bdiag_bin = np.zeros((N, N))
# for block in range(m4.num_blocks):
#     aux_matrix = np.mat(blocks == block).T * np.mat(blocks == block)
#     bdiag_bin += aux_matrix
#
# bdiag_bin = np.float64(bdiag_bin)
# Kbdiag = tf.linalg.matmul(full_cov, bdiag_bin)
#
# plt.imshow(Kbdiag)
# plt.show()
#
#
# kuf = Kuf(m4.inducing_variable, m4.kernel, X_data)
# kuu = Kuu(m4.inducing_variable, m4.kernel, jitter=default_jitter())
#
# Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
# V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf
#
# bdiagQff = tf.linalg.matmul(tf.transpose(V), V)
#
# bdiagQff = tf.linalg.matmul(bdiagQff, bdiag_bin)
#
#
# nu = Kbdiag - bdiagQff + tf.eye(X_data.shape[0], dtype=np.float64) * m4.likelihood.variance
#
# nuinv = tf.linalg.inv(nu)
#
# Baux = tf.linalg.matmul(V, nuinv)
# B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
#     Baux, V, transpose_b=True
# )
# L = tf.linalg.cholesky(B)
#
# beta = tf.linalg.matmul(nuinv, err)  # size [N, R]
# alpha = tf.linalg.matmul(V, beta)  # size [N, R]
#
# gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]
#
#
########################################################
# # Test PI(T)C "construct_blocks()" method
# from sklearn.cluster import KMeans
# # # Idea: return sparse binary matrices for each group of blocks
# block_sizes=None
# X_data, _ = m5.data
# dim = X_data.shape[1]
# num_blocks = int(X_data.shape[0] / m5.inducing_variable.num_inducing) # REVIEW:
# km = KMeans(n_clusters=num_blocks)
# km.fit(X_data)
# km.labels_
# blocks = km.labels_

########################################################
# # Test PIC "predict_f()" method
# full_cov = True
# Xnew = Xtest
# _, _, Luu, L, alpha, beta, gamma = m5.common_terms()
# Kus = Kuf(m5.inducing_variable, m5.kernel, Xnew)  # [M, N]
# w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N] => w^T w = Qss = Kus^T kuu^-1 Kus
#
#
# tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False) # => tmp^T tmp = gamma^T B^-1 gamma
# mean = tf.linalg.matmul(w, tmp, transpose_a=True) + m5.mean_function(Xnew) # => w -> sqrt(Q*n); tmp -> sqrt(Q*n)*[Kn+sigma^2*I]^-1*y
# intermediateA = tf.linalg.triangular_solve(L, w, lower=True) # intA^T intA = w^T B^-1 w = sqrt(Noise Variance)
#
# # Calculate K*n of PIC
# # Precompute K*n and Q*n so that for each test point it is just joining
# # matrix columns
# # When not in the same block
# #   w -> sqrt(Q*n)
# #   tmp -> sqrt(Q*n)*[Kn+sigma^2*I]^-1*y
# # When in the same block
# #   aux -> (w^2)^-1 -> (Qn)^-1
# #   aux(2) -> aux * w^T * tmp -> [Kn+sigma^2*I]^-1*y
# #   K*n -> Kus
# aux = tf.linalg.inv(tf.linalg.matmul(w, w, transpose_a=True))
# aux_2 = tf.linalg.matmul(aux, tf.linalg.matmul(w, tmp, transpose_a=True))
#
# # Create matrix with mix of K*n and Q*n
# # Call function which takes vector [Nxp] of points and returns number of block where each point (Nx1) belongs
# #   For vector of test points [Txp]
# #   For vector of inducing points [Uxp]
# # Create sparse binary matrix [TxU] to see where each point coincidence is present
# test_blocks = m5.km.predict(Xnew)
# sparse_blocks = m5.km.predict(X_ind)
# bdiag_bin_K = np.mat([test_blocks[i]==sparse_blocks for i in range(len(Xnew))])
#
# Kpitc = (bdiag_bin_K.T - w) * Kus + w
#
# mean = tf.linalg.matmul(Kpitc, aux_2)
#
# if full_cov:
#     var = (
#             m5.kernel(Xnew)
#             - tf.linalg.matmul(w, Kpitc, transpose_a=True)
#             + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
#     )
#     var = tf.tile(var[None, ...], [m5.num_latent_gps, 1, 1])  # [P, N, N]
# else:
#     var = (
#             m5.kernel(Xnew, full_cov=False)
#             - tf.reduce_sum(tf.matmul(w, Kpitc), 0)
#             + tf.reduce_sum(tf.square(intermediateA), 0)
#     )  # [N, P]
#     var = tf.tile(var[:, None], [1, m5.num_latent_gps])

# mean, var



# ######################################################
# # Test FITC "common_terms()" method
# start=timer()
# X_data, Y_data = m3.data
# m3_num_inducing = m3.inducing_variable.num_inducing
# m3_err = Y_data - m3.mean_function(X_data)  # size [N, R]
# m3_Kdiag = m3.kernel(X_data, full_cov=False)
# m3_kuf = Kuf(m3.inducing_variable, m3.kernel, X_data) # Covariance matrix bw inducing points (u) and training (f)
# m3_kuu = Kuu(m3.inducing_variable, m3.kernel, jitter=default_jitter()) # Covariance matrix bw inducing points (u)
#
# m3_Luu = tf.linalg.cholesky(m3_kuu)  # => Luu Luu^T = kuu
# m3_V = tf.linalg.triangular_solve(m3_Luu, m3_kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf
#
# m3_diagQff = tf.reduce_sum(tf.square(m3_V), 0)
# m3_nu = m3_Kdiag - m3_diagQff + m3.likelihood.variance # diag(K_f - Qff) + sigma^2 I
#
# m3_B = tf.eye(m3_num_inducing, dtype=default_float()) + tf.linalg.matmul(
#     m3_V / m3_nu, m3_V, transpose_b=True
# )
# m3_L = tf.linalg.cholesky(m3_B)
# m3_beta = m3_err / tf.expand_dims(m3_nu, 1)  # size [N, R]
# m3_alpha = tf.linalg.matmul(m3_V, m3_beta)  # size [N, R]
#
# m3_gamma = tf.linalg.triangular_solve(m3_L, m3_alpha, lower=True)  # size [N, R]
# end=timer()
# print(end-start)

#
# ######################################################
# # m4.maximum_log_likelihood_objective()
# err, nu, Luu, L, alpha, beta, gamma = m4.common_terms()
# nuinv = tf.linalg.inv(nu)
# aux = tf.linalg.matmul(nuinv,tf.square(err))
#
# mahalanobisTerm = -0.5 * tf.reduce_sum(aux) + 0.5 * tf.reduce_sum(tf.square(gamma))
# constantTerm = -0.5 * m4.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
# # log(nu) returns inf values (not valid or reduce_sum) -> take diagonal part
# logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(
#                                             tf.linalg.diag_part(nu)
#                                           )) \
#                      - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
# logNormalizingTerm = constantTerm + logDeterminantTerm
#
# mahalanobisTerm + logNormalizingTerm * m4.num_latent_gps
#
# ######################################################
# # m3.maximum_log_likelihood_objective()
# m3_err, m3_nu, m3_Luu, m3_L, m3_alpha, m3_beta, m3_gamma = m3.common_terms()
#
# m3_mahalanobisTerm = -0.5 * tf.reduce_sum(
#             tf.square(m3_err) / tf.expand_dims(m3_nu, 1)
#         ) + 0.5 * tf.reduce_sum(tf.square(m3_gamma))
#
# m3_constantTerm = -0.5 * m3.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
# m3_logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(m3_nu)) - tf.reduce_sum(
#     tf.math.log(tf.linalg.diag_part(m3_L))
# )
# m3_logNormalizingTerm = m3_constantTerm + m3_logDeterminantTerm
#
# m3_mahalanobisTerm + m3_logNormalizingTerm * m3.num_latent_gps

