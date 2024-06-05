import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from skimage.data import chelsea
from os import system

np.random.seed(1410)

# Data
img = chelsea()
img = img[::2,::2]

# Positional X
pX = np.meshgrid(np.linspace(-.75,.75,img.shape[1]), np.linspace(-.75,.75,img.shape[0]))
pX = np.array(pX).reshape(2,-1).T
cX = img.reshape(-1,3)

# Positional probe
qq = 256
dX = np.meshgrid(np.linspace(-1.25,1.25,qq), np.linspace(-1.25,1.25,qq))
dX = np.array(dX).reshape(2,-1).T
accu = np.zeros((qq,qq,3)) # Accumulator

# Model
max_iter = 500
model = MLPRegressor(hidden_layer_sizes=(100,100,100), 
                      max_iter=max_iter,
                      random_state=1410,
                      learning_rate_init=.1,
                      activation='relu')

for iter in range(max_iter):
    model.partial_fit(pX, cX)
    pp = model.predict(dX)
    
    r_img = pp.reshape((qq,qq,3))
    r_img -= np.min(r_img)
    r_img /= np.max(r_img)
        
    print(f'Iteration {iter+1}/{max_iter}')

    # Figure
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].imshow(img)
    ax[0,1].imshow(r_img)

    accu += r_img
    a = (accu-np.min(accu))/(np.max(accu)-np.min(accu))
    ax[1,0].imshow(a)
    
    ax[1,1].plot(model.loss_curve_)
    ax[1,1].grid(ls=":")
    ax[1,1].set_xlabel('Iteration')
    ax[1,1].set_ylabel('Loss')
    
    plt.savefig('foo.png')
    system('cp foo.png bar.png')
    plt.savefig('frames/nn9_%04i.png' % iter)
    
    plt.close()