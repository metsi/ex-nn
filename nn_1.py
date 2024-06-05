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
max_iter = 100
model = MLPClassifier(hidden_layer_sizes=(100), 
                      max_iter=max_iter,
                      random_state=1410,
                      learning_rate_init=0.001,)

# Training
for iter in range(max_iter):
    model.partial_fit(X, y, [0,1])
    
    y_pred = model.predict(X)
    pp = model.predict_proba(X)
    c = model.predict(M)
    cp = model.predict_proba(M)[:,1]
    score = accuracy_score(y, y_pred)
    
    print(f'Iteration {iter+1}/{max_iter}', score)

    # Figure
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].scatter(*X.T, c=y, cmap='coolwarm')
    ax[0,0].set_title('Class distribution')
    ax[0,0].grid(ls=":")
    
    ax[0,1].scatter(*X.T, c=pp[:,1], cmap='coolwarm')
    ax[0,1].set_title(f'Accuracy: {score:.2f}')
    ax[0,1].grid(ls=":")
    
    ax[1,0].scatter(*M.T, c=c, alpha=1, s=8, cmap='coolwarm')
    ax[1,0].set_title('Decision boundary')
        
    ax[1,1].plot(model.loss_curve_)
    ax[1,1].set_title('Loss curve')
    ax[1,1].grid(ls=":")
    ax[1,1].set_xlabel('Iteration')
    ax[1,1].set_ylabel('Loss')
    

    plt.savefig('foo.png')
    plt.savefig('frames/nn1_%04i.png' % iter)
    
    plt.close()