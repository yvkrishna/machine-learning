import numpy as np
import math

def sigmoid(x):
  value=1/(1+np.exp(-x))
  return value

def neuralNet(inputs,layers,num_of_outputs):
  theta=[];a=[];outputs=[]
  inputs=np.insert(inputs, 0, 1, axis=0)
  for i in range(0,layers+1):
    theta.append(np.random.rand((len(inputs)-1),(len(inputs))))
    if i==0:
      sig=np.matmul(theta[i], inputs.transpose())
    else:
      sig=np.matmul(theta[i], a[i-1].transpose())
    abc=sigmoid(sig)
    abc=np.insert(abc, 0, 1, axis=0)
    a.append(abc)
  theta_final=np.random.rand(num_of_outputs,len(inputs))
  sig2=np.matmul(theta_final, a[layers].transpose())
  outputs=sigmoid(sig2);
  return outputs;

outcome=neuralNet([2,3,4],1,1)
print(outcome)