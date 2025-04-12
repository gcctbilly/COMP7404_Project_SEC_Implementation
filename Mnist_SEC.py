# The SEC without local regression for Mnist dataset
from SEC_Model import SEC, sec_out_of_sample_extension, param_tuning_sec,clustering_accuracy
# load mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X[:5000] # shape (5000, 784)
y = y[:5000].astype(int) # shape (5000,)

print(X.shape, y.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=100)

X = pca.fit_transform(X)

# Split into in-sample/out-of-sample

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_train = X_train.T  # shape (784, n_train)
X_test = X_test.T    # shape (784, n_test)

X_train = np.array(X_train, dtype=np.float64)
X_test = np.array(X_test, dtype=np.float64)

print(X_train.shape, y_train.shape)

# 4. Tune on the training set
# 4.1 Wihtout local regression
param_grid_mu = [1e-9, 1e-6, 1e-3,1e0, 1e3]
param_grid_gammag = [1e-6, 1e-3, 1e0, 1e3]
param_grid_gammal = [1e-6, 1e-3, 1e0, 1e3]
k = [3,5,7]
c = 10
best_params, best_acc = param_tuning_sec(
    X_train,
    y_train,
    c=c,
    param_grid_mu=param_grid_mu,
    param_grid_gammag=param_grid_gammag,
    param_grid_gammal=[1.0],  # fix gamma_l if we're not using local
    k=k,
    normalized=True,
    use_local_regression=False  # not using local regression for now
)
print("Best params (mu, gamma_g, gamma_l, k):", best_params)
print("Best training (in-sample) accuracy:", best_acc)

# 5. Retrain with best parameters on the entire training set
mu_val, gg_val, gl_val,k = best_params
model = SEC(
    use_local_regression=False,
    k=k,
    mu=mu_val,
    gamma_g=gg_val,
    gamma_l=gl_val,
    normalized=True
)
labels_train_pred, F, W, b = model.fit(X_train, c)
train_acc_final = clustering_accuracy(labels_train_pred, y_train)
print("Final in-sample accuracy with best params:", train_acc_final)

#Out-of-sample extension on X_test
n_test = X_test.shape[1]
labels_test_pred = np.zeros(n_test, dtype=int)
for i in range(n_test):
    x_new = X_test[:, i]
    labels_test_pred[i] = sec_out_of_sample_extension(x_new, W, b)

test_acc = clustering_accuracy(labels_test_pred, y_test)
print("Out-of-sample (test) accuracy:", test_acc)