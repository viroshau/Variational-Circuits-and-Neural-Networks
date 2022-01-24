import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cvxgraphalgs as cvxgr
from scipy.optimize import minimize
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
            G[i][j]['weight'] = np.random.rand() #Assign edge weights to be a float between [0,1]
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
    """Returns the adjacency matrix as a numpy 2D matrix

    Args:
        G ([nx.graph]): [The graph instance that we want the adjacecy matrix of]

    Returns:
        [np.ndarray(len(G.nodes),len(G.nodes))]: [A 2D numpy array containing the edges of the matrix]
    """
    adjacencymatrix = np.zeros((len(G.nodes),len(G.nodes)))
    for i,j,w in G.edges(data = True):
        adjacencymatrix[i,j] = w['weight'] 
        adjacencymatrix[j,i] = w['weight']
    return adjacencymatrix

def EvaluateCutValue(G,x):
    """Evaluates the value of the cut given a bitstring x on graph G

    Args:
        G ([nx.graph]): [Graph instance to consider]
        x ([np.array): [1D array containing the bitstring at each position. The elements are either 1 or -1]
    
    Return:
        ([float]): The value of the cut given the configuration.
    """
    AdjacencyMatrix = CreateAdjacencyMatrix(G)
    return 0.25*np.sum(AdjacencyMatrix,axis = (0,1)) - 0.25*x@AdjacencyMatrix@x

def EvaluateCutValueDifferentversion(x,G):
    cost = 0 
    for i,j,w in G.edges(data = True):
        cost -= 0.5*w['weight']*(1-x[i]*x[j])
    return cost

def EvaluateCutOnDataset(dataset,G):
    cost = 0 
    for k in range(len(dataset)):
        cost += EvaluateCutValueDifferentversion(dataset[k],G)
    return cost/1000

def DrawGraph(G):
    """Draws the graph G

    Args:
        G ([nx.graph]): Graph to be drawn
    """
    pos = nx.bipartite_layout(G,nx.bipartite.sets(G)[0])
    #pos = nx.shell_layout(G)
    plt.figure()
    nx.draw(G,pos)

    edgelabels = {}
    for i,j,w in G.edges(data = True):
        edgelabels[(i,j)] = round(w['weight'],3)

    nx.draw_networkx_edge_labels(G,pos,edge_labels= edgelabels)
    nx.draw_networkx_labels(G,pos = pos)
    plt.show()

def CreateBatchOfGraphs(numGraphs,):
    return

def performScipyOptimizationProcedure(init_params,cost_h):
    dev2 = qml.device('default.qubit', wires=nqubits) #need to use the torch default qubit instead of the usual default.qubit in order to take in G as a variable.
    
    @qml.qnode(dev2)
    def circuit(x,cost_h):
        for i in range(nqubits):
            qml.Hadamard(wires = i)
        
        p = len(x)//2
        for i in range(p):
            for j,k in G.edges():
                qml.CNOT(wires = [j,k])
                qml.RZ(-x[:p][i], wires = k)
                qml.CNOT(wires = [j,k])

            #for j,k,w in edges:
                #qml.CNOT(wires = [int(j.item()),int(k.item())])
                #qml.RZ(-x[:p][i], wires = int(k.item()))
                #qml.CNOT(wires = [int(j.item()),int(k.item())])

            for j in range(nqubits):
                qml.RX(2*x[p:][i],wires = j)

        return qml.expval(cost_h.item())

    optimizer = minimize(circuit, init_params, args = (cost_h), method='BFGS', jac = qml.grad(circuit, argnum=0))
    return optimizer
