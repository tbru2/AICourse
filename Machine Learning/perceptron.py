import numpy as np

data = np.loadtxt('input1.csv', delimiter=",")
weights = np.zeros(3)
prevWeights = np.array([float("inf") for i in range(len(weights))])
fp = open("output1.csv", 'w')

cols = np.shape(data)[0]
data = np.concatenate((np.ones((cols,1)), data), axis=1)

while 1:
    
    for row in data:
        a = np.dot(row[:3], weights)
        a = np.where(a > 0, 1, -1)
        if a * row[3] <= 0:
            
            weights += row[3] * row[:3] 
            
    fp.write(str(weights[1]) + ',' + str(weights[2])  + ',' + str(weights[0]) + '\n')
   
    if np.array_equal(weights - prevWeights,np.zeros(len(weights))):
        break
    prevWeights = np.copy(weights)
  
fp.close()