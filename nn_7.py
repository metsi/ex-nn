import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from methods import complex_set

np.random.seed(1410)

# Data
X, y = complex_set()
y = np.eye(2)[y.astype(int)]

# Random projection of X
X_r = np.dot(X, np.random.randn(X.shape[1], 2))

# Model
max_iter = 100
model = MLPRegressor(hidden_layer_sizes=(3,3,3), 
                      max_iter=max_iter,
                      random_state=1410,
                      learning_rate_init=0.01,)

# Training
for iter in range(max_iter):
    model.partial_fit(X, y)
    pp = model.predict(X)
        
    print(f'Iteration {iter+1}/{max_iter}')

    # Figure
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].scatter(*X_r.T, c=y[:,1], cmap='coolwarm')
    ax[0,0].set_title('Class distribution')
    ax[0,0].grid(ls=":")
        
    ax[0,1].scatter(*pp.T, c=y[:,1], cmap='coolwarm')
    ax[0,1].grid(ls=":")
    ax[0,1].set_xticks(np.linspace(0,1,11))
    ax[0,1].set_yticks(np.linspace(0,1,11))
    ax[0,1].set_title('Support distribution')
    ax[0,1].plot([0,1],[0,1], c='black', ls='--')

    ax[1,0].plot(model.loss_curve_)
    ax[1,0].set_yscale('log')
    ax[1,0].set_title('Log loss curve')
    ax[1,0].grid(ls=":")
    ax[1,0].set_xlabel('Iteration')
    ax[1,0].set_ylabel('Loss')
    
    ax[1,1].plot(model.loss_curve_)
    ax[1,1].set_title('Loss curve')
    ax[1,1].grid(ls=":")
    ax[1,1].set_xlabel('Iteration')
    ax[1,1].set_ylabel('Loss')
    
    plt.savefig('foo.png')
    plt.savefig('frames/nn7_%04i.png' % iter)
    
    plt.close()