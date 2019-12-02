import numpy as np
import math

theta=np.random.rand(3,4)
print(theta)
inputs=np.array([1,2,3,4],int)

 sig=np.matmul(theta, inputs.transpose())
 
 def sigmoid(x):
  value=1/(1+np.exp(-x))
  return value
a=sigmoid(sig)

a = np.insert(a, 0, 1, axis=0)
print(a)

theta2=np.random.rand(1,4)
sig2=np.matmul(theta2, a.transpose())
print(sig2)

output=sigmoid(sig2)
print(output)
