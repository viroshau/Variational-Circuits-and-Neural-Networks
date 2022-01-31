import numpy as np
import matplotlib.pyplot as plt


percentageimprovement = np.zeros((2,8))

means = np.zeros((8,2,2))
variances = np.zeros((8,2,2))

for p in range(1,9):
    for i in range(0,2):
        if i == 0:
            results = np.load(f'ESCAPEResults/ADAM,p={p},SGD_NN_lr=0.05,Adam_VQC_lr=0.3,M=80,T=350,QAOA_iter = 60.py.npy')
        else:
            results = np.load(f'ESCAPEResults/ADAM,p={p},SGD_NN_lr=0.05,Adam_VQC_lr=0.3,M=80,T=350,QAOA_iter = 60,{1}.py.npy')
        results0 = results[0]
        results1 = results[1]
        diff = results0 - results1
        #print(len(diff[diff > 0.1]))
        percentageimprovement[i,p-1] = len(diff[diff > 0.1])
        means[p-1,i,0] = np.mean(results0)
        means[p-1,i,1] = np.mean(results1)
        variances[p-1,i,0] = np.var(results0)
        variances[p-1,i,1] = np.var(results1)

print(variances)

plt.figure()
plt.title('Percentage of initializations where a costimprovement of 0.1 was discovered')
plt.ylabel('% improvement')
plt.xlabel('p')
plt.hist(x = [1,2,3,4,5,6,7,8],bins = 8,weights = np.sum(percentageimprovement,axis = 0)/200 )
#plt.show()

