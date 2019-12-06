import numpy as np
import math

def sigmoid(x):
  value=1/(1+np.exp(-x))
  return value

def neuralNet(inputs,outputs,layers,no_of_units):
  theta=np.random.rand(len(inputs),len(inputs)+1)
  inputs=np.insert(inputs, 0, 1, axis=0)
  sig=np.matmul(theta, inputs.transpose())
  a=sigmoid(sig)
  a = np.insert(a, 0, 1, axis=0)
  

theta2=np.random.rand(1,4)
sig2=np.matmul(theta2, a.transpose())
print(sig2)
