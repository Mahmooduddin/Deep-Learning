import numpy as np
a=np.zeros((2,2))
b=np.ones((2,2))
print(np.sum(b,axis=1))
print(a.shape)

print(a)
print(np.reshape(a,(1,4)))