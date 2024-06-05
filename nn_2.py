import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from methods import simple_set, mgrid

np.random.seed(1410)

# Data
X, y = simple_set()
M = mgrid()

# Model
max_iter = 1000
model = MLPClassifier(hidden_layer_sizes=(100), 
                      max_iter=max_iter,
                      random_state=1410,
                      learning_rate_init=0.001,)

# Training
for iter in range(max_iter):
    model.partial_fit(X, y, [0,1])
    y_pred = model.predict(X)
    c = model.predict(M)
    xpp = model.predict_proba(X)    
    pp = model.predict_proba(M)
    cp = pp[:,1]
    
    score = accuracy_score(y, y_pred)
    
    print(f'Iteration {iter+1}/{max_iter}', score)

    # Figure
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].scatter(*X.T, c=y, cmap='coolwarm')
    ax[0,0].set_title('Class distribution')
    ax[0,0].grid(ls=":")
        
    ax[0,1].hist(xpp[y==0][:,1], bins=32, color='red', alpha=.5)
    ax[0,1].hist(xpp[y==1][:,1], bins=32, color='blue', alpha=.5)
    ax[0,1].set_xlim(0,1)
    ax[0,1].grid(ls=":")
    ax[0,1].set_xticks(np.linspace(0,1,11))
    ax[0,1].set_title('Support distribution')
        
    ax[1,0].scatter(*M.T, c=c, alpha=1, s=8, cmap='coolwarm')
    ax[1,0].set_title('Decision boundary')
    ax[1,1].scatter(*M.T, c=cp, alpha=1, s=8, cmap='coolwarm')
    ax[1,1].set_title('Decision probability')

    plt.savefig('foo.png')
    plt.savefig('frames/nn2_%04i.png' % iter)
    
    plt.close()