import numpy as np
import sys

def lin_reg(inputs, targets, alpha, fp):
    
    sum = 0.
    sumZ = 0.
    j = 0
    k = 0
    inputs = np.concatenate((np.ones((np.shape(inputs)[0],1)), inputs), axis=1)
    beta = np.zeros(np.shape(inputs)[1])
   
    for i in range(1,101):
        for row in inputs:
            sum += np.dot(row, np.transpose(beta)) - np.array([targets[j], targets[j], targets[j]])
            sumZ += sum * row
            j+=1
        beta[0] -= (alpha/ np.shape(inputs)[0]) * sum[0]
        beta[1:] -= ((alpha/ np.shape(inputs)[0]) * sumZ[1:])
        sum = 0.
        sumZ = 0.
        j = 0
    
    fp.write(str(alpha) +',' + str(i) + ',')
    for i in range(np.shape(beta)[0]-1):
        fp.write(str(beta[i]) + ',')
    fp.write(str(beta[np.shape(beta)[0] - 1]) + '\n') 



def main():
    fp = open("output2.csv", 'w')
    data = np.loadtxt('input2.csv', delimiter=',')
    #data = (data - np.mean(data))/np.std(data)
    inputs = data[:,:2]
    inputs = (inputs - np.mean(inputs,axis=0))/np.std(inputs, axis=0)
    targets = data[:,2]
    alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 20.]
    for param in alpha:
        lin_reg(inputs, targets, param, fp)
    fp.close()
    
if __name__ == '__main__':
    main()