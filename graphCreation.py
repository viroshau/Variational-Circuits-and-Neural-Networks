import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cvxgraphalgs as cvxgr
from scipy.optimize import minimize
from tqdm import tqdm
import torch

#The graphs should be a nx graph where they have a weight attribute.

def CreateRenyiGraph(n,p,randomweights = False,seed = None):
    G = nx.fast_gnp_random_graph(n,p,seed = seed)
    if randomweights == True:
        for i,j in G.edges:
            G[i][j]['weight'] = np.random.rand() #Assign edge weights to be a float between [0,1]
    else:
        for i,j in G.edges:
            G[i][j]['weight'] = 1 #Assign weights to be 1 if you don't want weights
    return G

def CreateRegularGraph(n,degree,randomweights = False,seed = None):
    G = nx.generators.random_regular_graph(degree,n,seed = seed)
    if randomweights == True:
        for i,j in G.edges:
            G[i][j]['weight'] = np.random.standard_normal() #Assign edge weights to be a float between [0,1]
    else:
        for i,j in G.edges:
            G[i][j]['weight'] = 1 #Assign weights to be 1 if you don't want weights
    return G

def CreateGraphFromList(list):
    #List contains tuples where the elements of the tuple are (i,j,w) where i,j are nodes and w is the weight between the nodes
    G = nx.Graph()
    for i,j,w in list:
        G.add_edge(i,j,weight = w)
    return G

def BestClassicalHeuristicResult(G):
    cut_value_classical = 0
    left = {}
    right = {}
    for i in range(10):
        #Perform the Classical routine 10 times in order to get higher probability of getting the maxiumim cut.
        recovered = cvxgr.algorithms.goemans_williamson_weighted(G)
        cut_value = recovered.evaluate_cut_size(G)
        if cut_value > cut_value_classical:
            cut_value_classical = max(cut_value_classical,cut_value)
            left = recovered.left
            right = recovered.right
        #print(recovered.left,recovered.right,cut_value)
    print(f'The best classical cost is: {cut_value_classical}. Left = {left}, Right = {right}')
    return cut_value_classical,left,right

def CreateAdjacencyMatrix(G):
    """Returns the adjacency matrix as a numpy 2D matrix. This utilizes for-loops, however is only called once so it's fine

    Args:
        G ([nx.graph]): [The graph instance that we want the adjacecy matrix of]

    Returns:
        [np.ndarray(len(G.nodes),len(G.nodes))]: [A 2D numpy array containing the edges of the matrix]
    """
    adjacencymatrix = np.zeros((len(G.nodes),len(G.nodes)))
    for i,j,w in G.edges(data = True):
        adjacencymatrix[i,j] = w['weight'] 
        adjacencymatrix[j,i] = w['weight']
    return torch.tensor(adjacencymatrix,dtype = torch.float32)

def DrawGraph(G):
    """Draws the graph G

    Args:
        G ([nx.graph]): Graph to be drawn
    """
    #pos = nx.bipartite_layout(G,nx.bipartite.sets(G)[0])
    pos = nx.shell_layout(G)
    plt.figure()
    nx.draw(G,pos)

    edgelabels = {}
    for i,j,w in G.edges(data = True):
        edgelabels[(i,j)] = round(w['weight'],3)

    nx.draw_networkx_edge_labels(G,pos,edge_labels= edgelabels)
    nx.draw_networkx_labels(G,pos = pos)
    plt.show()

def EvaluateCutValue(adjacencymatrix,x):
    """Evaluates the value of the cut given a bitstring x on graph G

    Args:
        G ([nx.graph]): [Graph instance to consider]
        x ([np.array): [1D array containing the bitstring at each position. The elements are either 1 or -1]
    
    Return:
        ([float]): The value of the cut given the configuration.
    """
    return -1*(0.25*torch.sum(adjacencymatrix,dim = (0,1)) - 0.25*x@adjacencymatrix@x)

def EvaluateCutValueDifferentversion(x,G): 
    """The naive method of calculating the cut-value of a bitstring by iterating through the edges-list of graph G

    Args:
        x ([torch.tensor(N)]): [tensor of N components containing the N-bitstring of [1,-1]]
        G ([nx.graph]): [A graph instance at which to evaluate the cut-value of bitstring x]

    Returns:
        [float]: [returns the cut value of bitstring x]
    """
    cost = 0 
    for i,j,w in G.edges(data = True): 
        cost -= 0.5*w['weight']*(1-x[i]*x[j])
    return cost

def EvaluateCutOnDatasetBADVERSION(dataset,adjacencymatrix):
    """The naive method of calculating the average cost of a dataset of bitstrings. This is bad because of the use of foor-loops 
    which is not good when used with the pytorch framework

    Args:
        dataset ([torch.tensor(N_samples,len(G.nodes))]): [Dataset containign the bitstrings where each component/row has elements 1,-1]
        adjacencymatrix ([torch.tensor(N.nodes,N.nodes)]): [adjacencvy matrix of graph G]

    Returns:
        [float]: [returns the average cost]
    """
    cost = 0
    #evaluate them all at the same time, compute the mean
    for k in range(len(dataset)):
        cost += EvaluateCutValue(adjacencymatrix,dataset[k])
    return cost/len(dataset)

def EvaluateCutOnDataset(dataset,adjacencymatrix):
    """Given a dataset filled with strings of type [1,-1,1,1,-1,-1] etc, this functions will evalute the mean cost of the set of strings

    Args:
        dataset ([torch(N_samples,len(G.nodes))]): [The dataset of pauli-Z eigenvalue strings]
        adjacencymatrix ([type]): [The adjacency matrix of graph G]

    Returns:
        [ndarray]: [Returns the cost of all N_samples in the dataset]
    """
    y = torch.matmul(adjacencymatrix, dataset.T).T #Perform adjacent multiplication on all vectors in dataset. returnsize = (N_samples,nodes)
    xy = (torch.sum(dataset*y,dim = 1)) # perform xAx on all N_samples
    result = 0.25*xy - 0.25*torch.sum(adjacencymatrix,dim = (0,1)) 
    return result 

def CreateBinaryList(numqubits):
    """Create a list of list containing all binary configurations that can exist with a given number of qubits

    Args:
        numqubits ([int]): [Number of qubits that can be measured]

    Returns:
        [type]: [python 2D list where each row is ]
    """
    return [[int(i) for i in f'{j:0{numqubits}b}'] for j in range(2**numqubits)]

def listofBinaryToInt(binarylist):
    descimalnumber = 0
    for i in range(len(binarylist)):
        if binarylist[len(binarylist)-i-1] == 1:
            descimalnumber += 2**i
    return descimalnumber

