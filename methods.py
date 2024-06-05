import numpy as np

def simple_set(spc=100):
    X = np.concatenate((
        np.random.normal(-1, .25, (spc, 2)),
        np.random.normal( 1, .25, (spc, 2))
    ))
    y = np.concatenate((np.zeros(spc), np.ones(spc)))
    return X, y

def complex_set(spc=100, random_features=100):
    X = np.concatenate((
        np.random.normal(-1, .25, (spc, 2)),
        np.random.normal( 1, .25, (spc, 2)),
    ))
    X = np.concatenate((X, 
                        np.random.normal(0, 5, (2*spc, random_features))), axis=1)
    y = np.concatenate((np.zeros(spc), np.ones(spc)))
    return X, y

def mgrid(r=2, q=128):
    X = np.meshgrid(np.linspace(-r, r, q), np.linspace(-r, r, q))
    X = np.array(X).reshape(2, -1).T
    return X
    
    
    