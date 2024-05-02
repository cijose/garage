import numpy as np
import scipy.io
data = scipy.io.loadmat('features.mat')["descriptors"]

num_classes = 632
m = 316
dim = data.shape[1]

gal_feats = data[:num_classes, :]
prob_feats = data[num_classes:, :]

p = np.random.permutation(np.arange(num_classes))

gal_train = gal_feats[p[:m]]
prob_train = prob_feats[p[:m]]

gal_test = gal_feats[p[m:]]
prob_test = prob_feats[p[m:]]

xtrain = np.zeros((m, 2, dim))
for i in range(m):
    xtrain[i, 0, :] = gal_train[i, :]
    xtrain[i, 1, :] = prob_train[i, :]

xtest = np.zeros((m, 2, dim))
for i in range(m):
    xtest[i, 0, :] = gal_test[i, :]
    xtest[i, 1, :] = prob_test[i, :]

print(xtrain.shape)
print(xtest.shape)



