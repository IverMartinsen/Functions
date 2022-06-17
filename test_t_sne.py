import numpy as np
from t_sne import t_sne

sample_size = 100
dimensionality = 50

X = np.random.normal(size=(sample_size, dimensionality))

perplexity = 30

epochs = 100

def test_t_sne():

    return t_sne(X, perplexity, epochs)

if __name__=='__main__':
    test_t_sne()