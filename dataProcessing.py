import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphCreation import *

seed = 51
np.random.seed(seed) #Use seed to generate a random graph
G = CreateRegularGraph(5,4,True,seed = seed)
#G = CreateGraphInstanceB(8,5,seed = seed)
np.random.seed() #Remove the random seed when generating initial points, etc
adjacencymatrix = CreateAdjacencyMatrix(G)
clEnergy,left,right = BestClassicalHeuristicResult(G)

sns.set_style("whitegrid")

QAOAresults = []
ESCAPEresults = []

simulations = 4
percentageimprovement = np.zeros(simulations)
filestring = f'SHOTBASED5NodeESCAPEResultsSameGraph{seed}'
for i in range(1,simulations+1):
    #results1 = np.load(f'ESCAPEResultsSameGraph47/ADAM,p={i},SGD_NN_lr=0.05,Adam_VQC_lr=0.3,M=80,T=350,QAOA_iter = 60.py.npy')
    results1 = np.load(filestring + f'/ADAM,p={i},init:150,NN:100,heaviside:20,final_iterations:150.npy')
    diff = results1[0] - results1[1]
    improvements = diff < 0.1
    percentageimprovement[i-1] = len(diff[improvements])
    QAOAresults.append(results1[0,improvements])
    ESCAPEresults.append(results1[1,improvements])

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,7.5))

bplot1 = ax1.boxplot(QAOAresults, positions = np.array(list(range(simulations)))*2-0.3,patch_artist=True)
for j in bplot1['boxes']:
        j.set_facecolor('lightblue')
bplot2 = ax1.boxplot(ESCAPEresults, positions = np.array(list(range(simulations)))*2+0.3,patch_artist=True)
for j in bplot2['boxes']:
        j.set_facecolor('lightgreen')

ax1.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ["QAOA", "ESCAPE"], loc='upper right')
ax1.set_xticks([2*i for i in range(simulations)])
ax1.set_xticklabels(list(range(1,simulations+1)))
ax1.set_xlabel('p')
ax1.set_ylabel('Cost')
ax1.set_title('Given improvement, plot the new cost')

ax2.hist(x = list(range(1,simulations+1)),bins = simulations,weights = percentageimprovement)
ax2.set_xlabel('p')
ax2.set_ylabel('% improvement')
ax2.set_title('Percentage of cases where cost-improvement was over 0.1')

plt.show()