import GPflow.gpflow as gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GPflow.gpflow.config import default_float, default_jitter
from GPflow.gpflow.utilities import print_summary, set_trainable, to_default_float
from GPflow.gpflow.models import maximum_log_likelihood_objective, training_loss_closure
from GPflow.gpflow.ci_utils import ci_niter

from GPflow.gpflow.covariances.dispatch import Kuf, Kuu


def plot(model, color, ax, lims=[-1,11], show_xs=True, show_ind_locations=True):
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
_ = plt.plot(X, Y, "kx", mew=2)
plt.show()

data = (
    tf.convert_to_tensor(X, dtype=default_float()),
    tf.convert_to_tensor(Y, dtype=default_float()),
)
X_ind = X[0:N-1:int(N/N_ind)]
inducing_variable = tf.convert_to_tensor(X_ind, dtype=default_float())

# Create models
kernel = gpflow.kernels.Matern32()

m1 = gpflow.models.GPR(data, kernel=kernel)

m2 = gpflow.models.SGPR(
    data, kernel=kernel, inducing_variable=inducing_variable
)
set_trainable(m2.inducing_variable, False)

m3 = gpflow.models.GPRFITC(
    data, kernel=kernel, inducing_variable=inducing_variable
)
set_trainable(m3.inducing_variable, False)

m4 = gpflow.models.GPRPITC(
    data, kernel=kernel, inducing_variable=inducing_variable
)
set_trainable(m4.inducing_variable, False)

models = [m1,m2,m3,m4]

for model in models:
    print_summary(model)

# Optimize models
for model in models:
    print("Starting training of ", type(model).__name__)
    opt = gpflow.optimizers.Scipy() # BFGS by default
    loss_closure = training_loss_closure(model, data)
    opt.minimize(
        loss_closure,
        variables=model.trainable_variables,
        options=dict(maxiter=ci_niter(1000)),
        compile=True,
    )
    print("Finished training of ", type(model).__name__)

# Plot results
f, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
plot(m1, "C0", ax[0, 0], show_xs=True)
plot(m2, "C1", ax[0, 1], show_xs=True)
plot(m3, "C2", ax[1, 0], show_xs=True)
plot(m4, "C3", ax[1, 1], show_xs=False)
plt.show()

# #####################################
# KIN40K (d=9)
# #####################################

# Load data FIXME: remove skiprows
kin40k_toy = np.loadtxt("data/kin40k/kin40k_train_data.asc", skiprows=9000) # Toy example of kin40k
kin40k_toy_l = np.loadtxt("data/kin40k/kin40k_train_labels.asc", skiprows=9000).reshape(-1, 1) # Toy example of kin40k
kin40k_test_toy = np.loadtxt("data/kin40k/kin40k_test_data.asc", skiprows=29000) # Toy example of kin40k
kin40k_test_toy_l = np.loadtxt("data/kin40k/kin40k_test_labels.asc", skiprows=29000) # Toy example of kin40k
N = len(kin40k_toy)
N_t = len(kin40k_test_toy)
N_ind = len(kin40k_toy) * 0.01  # REVIEW: Proportion of inducing points?
n_dim = kin40k_toy.shape[1]

kin40k_data = (
    tf.convert_to_tensor(kin40k_toy, dtype=default_float()),
    tf.convert_to_tensor(kin40k_toy_l, dtype=default_float()),
)
kin40k_ind = kin40k_toy[0:N-1:int(N/N_ind)]
kin40k_ind = tf.convert_to_tensor(kin40k_ind, dtype=default_float())
data = kin40k_data


# Create models function for future use
def create_models(data,
                  kernel,
                  inducing_variable,
                  optimize=True,
                  opt= gpflow.optimizers.Scipy(),  # BFGS
                  niter = 1000
                  ):

    m1 = gpflow.models.GPR(data, kernel=kernel)

    m2 = gpflow.models.SGPR(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m2.inducing_variable, False)

    m3 = gpflow.models.GPRFITC(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m3.inducing_variable, False)

    m4 = gpflow.models.GPRPITC(
        data, kernel=kernel, inducing_variable=inducing_variable
    )
    set_trainable(m4.inducing_variable, False)

    models = [m1, m2] #FIXME:

    for model in models:
        print_summary(model)

    if optimize:
        # Optimize models
        for model in models:
            print("Starting training of ", type(model).__name__)
            loss_closure = training_loss_closure(model, data)
            opt.minimize(
                loss_closure,
                variables=model.trainable_variables,
                options=dict(maxiter=ci_niter(niter)),
                compile=True,
            )
            print("Finished training of ", type(model).__name__)

    return models


# https://github.com/GPflow/GPflow/issues/1606
# kin40k_m1, kin40k_m2, kin40k_m3, kin40k_m4 = create_models(kin40k_data, gpflow.kernels.Matern32(ndim=n_dim), kin40k_ind)
models = create_models(kin40k_data, gpflow.kernels.Matern32(lengthscales=np.ones(n_dim)), kin40k_ind)

# FIXME: check number of dimensions
def plot_models(models):
    nrow = int(np.floor(np.sqrt(len(models))))
    ncol = int(np.ceil(len(models)/nrow))
    f, ax = plt.subplots(nrow, ncol, figsize=(12, 9), sharex=True, sharey=True)
    for i,j in zip(range(nrow), range(ncol)):
        try:
            plot(models[i*nrow+j], "C"+str(i*nrow+j), ax[i, j], show_xs=True)
        except:
            plot(models[i*nrow+j], "C"+str(i*nrow+j), ax[j], show_xs=True)
    plt.show()

plot_models(models)



# #####################################
# TESTING PLAYGROUND
# #####################################
# # Test PITC "common_terms()" method
# # m4.common_terms()
# X_data, Y_data = m4.data
# num_inducing = m4.inducing_variable.num_inducing
# err = Y_data - m4.mean_function(X_data)  # size [N, R] = [1000, 1]
# blocks = []
# m4.construct_blocks()
# for block in m4.blocks:
#      k = m4.kernel(X_data[block[0]:block[1]], full_cov=True) # Each block matrix passed through Cov. fct.
#      blocks.append(k)
# Kbdiag = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
# Kbdiag = tf.linalg.LinearOperatorBlockDiag(Kbdiag).to_dense()
# # Plot image of block-diagonal matrix
# plt.imshow(Kbdiag)
# plt.show()
#
# kuf = Kuf(m4.inducing_variable, m4.kernel, X_data)
# kuu = Kuu(m4.inducing_variable, m4.kernel, jitter=default_jitter())
#
# Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
# V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf
#
# blocks=[]
# bdiagQff = tf.linalg.matmul(tf.transpose(V), V)
# for block in m4.blocks:
#      k = bdiagQff[block[0]:block[1], block[0]:block[1]]
#      blocks.append(k)
# bdiagQff = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
# bdiagQff = tf.linalg.LinearOperatorBlockDiag(bdiagQff).to_dense()
# # Plot block diagonal matrix
# plt.imshow(bdiagQff)
# plt.show()
#
# nu = Kbdiag - bdiagQff + tf.eye(X_data.shape[0], dtype=np.float64)*m4.likelihood.variance
#
# nuinv = tf.linalg.inv(nu)
#
# Baux = tf.linalg.matmul(V, nuinv)
# B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
#     Baux, V, transpose_b=True
# )
# L = tf.linalg.cholesky(B)
#
# beta = tf.linalg.matmul(nuinv, err) # size [N, R]
# alpha = tf.linalg.matmul(V, beta)  # size [N, R]
#
# gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]
#
#
# ######################################################
# # Test FITC "common_terms()" method
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
