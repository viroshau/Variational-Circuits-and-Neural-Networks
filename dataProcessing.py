import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns

sns.set_style("whitegrid")

QAOAresults = []
ESCAPEresults = []

percentageimprovement = np.zeros(8)

for i in range(1,9):
    #results1 = np.load(f'ESCAPEResultsSameGraph47/ADAM,p={i},SGD_NN_lr=0.05,Adam_VQC_lr=0.3,M=80,T=350,QAOA_iter = 60.py.npy')
    results1 = np.load(f'ESCAPEResultsSameGraph46/ADAM,p={i},SGD_NN_lr=0.05,Adam_VQC_lr=0.3,M=80,T=350,QAOA_iter = 60.py.npy')
    diff = results1[0] - results1[1]
    improvements = diff > 0.1
    percentageimprovement[i-1] = len(diff[improvements])
    QAOAresults.append(results1[0,improvements])
    ESCAPEresults.append(results1[1,improvements])

fig, (ax1, ax2) = plt.subplots(1, 2)

bplot1 = ax1.boxplot(QAOAresults, positions = np.array(list(range(8)))*2-0.3,patch_artist=True)
for j in bplot1['boxes']:
        j.set_facecolor('lightblue')
bplot2 = ax1.boxplot(ESCAPEresults, positions = np.array(list(range(8)))*2+0.3,patch_artist=True)
for j in bplot2['boxes']:
        j.set_facecolor('lightgreen')

ax1.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ["QAOA", "ESCAPE"], loc='upper right')
ax1.set_xticks([0,2,4,6,8,10,12,14])
ax1.set_xticklabels([1,2,3,4,5,6,7,8])
ax1.set_xlabel('p')
ax1.set_ylabel('Cost')
ax1.set_title('Given improvement, plot the new cost')


ax2.hist(x = [1,2,3,4,5,6,7,8],bins = 8,weights = percentageimprovement)
ax2.set_xlabel('p')
ax2.set_ylabel('% improvement')
ax2.set_title('Percentage of cases where cost-improvement was over 0.1')
plt.show()