import numpy as np

data = np.loadtxt('input1.csv', delimiter=",")
weights = np.zeros(3)

fp = open("output1.csv", 'w')

cols = np.shape(data)[0]
data = np.concatenate((np.ones((cols,1)), data), axis=1)

while 1:
    for row in data:
        f = np.sum(np.dot(np.transpose(row[:3]), weights))
        if f <= 0:
            f = -1
        else:
            f = 1
    
        for i in range(np.shape(weights)[0]):
            if f * row[3] <= 0:
                weights[i] += row[3] * row[i] 
        if weights[2] - weights[1] == 0:
            break
    fp.write(str(weights[1]) + ',' + str(weights[2])  + ',' + str(weights[0]) + '\n')
    if weights[2] - weights[1] == 0:
        break
        
fp.close()
        

        
