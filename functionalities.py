import numpy as np
from tqdm import tqdm 
import networkx as nx


def get_time_bounds(data,inf,sup):
    mi = data['t'].min()
    ma = data['t'].max()
    if mi < inf:
        inf = mi 
    if ma > sup:
        sup = ma
    return inf, sup


def get_time_interval(files, inf, sup):
    time_bounds = []
    for file in files:
        time_bounds.append(get_time_bounds(file, inf, sup))
    
    inf = time_bounds[0][0]
    sup = time_bounds[0][1]
    for i in range(len(time_bounds)):
        if time_bounds[i][0] < inf:
            inf = time_bounds[i][0]
        if time_bounds[i][1] > sup:
            sup = time_bounds[i][1]
    return inf, sup


def weighter(df):
    node_weights = dict(df['u'].value_counts()/(max(df['u'].value_counts())*2))
    #we assign to every node a weight that oscillates from 0 to 0.5, so the actions of the most
    #active user will have a maximum score of 1, and the other values will be lower until their initial weight.
    #We put all the values into a dictionary.
    df['weight'] = round(df['weight'] + df['u'].map(node_weights), 3)
    #all the values in the dictionary are assigned to the correspondent edge 
    df = df.dropna()
    #sometimes there could be errors that cause NaN values into the weight column, so we delete the respective rows
    return df


def centrality(G,v):
    return(G.degree(v))


def degree_centrality(G, v):
    return(G.degree(v) / G.number_of_edges())


def distance(G, start, end):
    explored = []
    queue = [[start]]
     
    if start == end:
        # print("start = end")
        return 0, [start]
     
    # Loop to vistit the nodes in the graph
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # if the current node is not yet visited
        # add it to the neighbours
        if node not in explored:
            neighbours = G[node]
             
            # iterate over the neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # stop if the neighbour node is the end node
                if neighbour == end:
                    # print("Shortest path = ", *new_path)
                    return len(new_path), new_path
            explored.append(node)
 
    # in this last case there is no path connecting node start and node end
    # so the distance is infinite (set to -1)
    # print("No connecting path between node start and node end")
    return -1, []


def closeness(G, u):
    sum_ = 0
    n = len(G)
    for v in tqdm(G.nodes()):
        dist, path = distance(G,u,v)
        if (dist!= -1 and (dist!=0)):
            sum_ += int(dist)
    if sum_>0: 
        return ((n-1)/sum_)
    else: 
        return 0
    
    
def betweeness(G,q):
    n = len(G)
    q_sum = 0
    tot = 0
    for u in tqdm(G.nodes):
        for v in (G.nodes):
            dist,path = distance(G,u,v) # or djikstra
            if dist!=0 and dist !=-1:
                tot += int(dist)
                if(q in path):
                    q_sum += int(dist)
    
    num = 2*q_sum/tot
    try:
        out = num / n*n - 3*n + 2
        return out
    except:
        print("Division by zero")
        
        
def page_rank(G, d = 0.85, iters = 10):
    N = len(G)
    # initialize the weighted rank for each node
    p_rank = dict.fromkeys(G.nodes(), 1/G.number_of_nodes())
    
    for i in range(iters):

        supp_rank = p_rank.copy()
        
        for page in G.nodes():
            predecessors =  list(G.predecessors(page))  # return set of pages(nodes) direct to page
            if len(predecessors) == 0:
                continue
            else:
                weighted = [ p_rank[i] for i in predecessors ] # return weighted rank of pages directing to page
                out_degree = [ G.out_degree(i, weight='weight')  for i in predecessors ] # return number of oubound links


                new_rank = ((1-d)/N)+(d*sum( np.array(weighted) / np.array(out_degree)))  # calculate the new weight rank for page -> the formula

                supp_rank[page] = new_rank  # update new weighted rank
                
        p_rank = supp_rank.copy()
    
    return p_rank 


def time_splitter(G, time_interval):
    len_step = round((time_interval[1] - time_interval[0])/4)
    
    G1 = nx.Graph()
    edges1 = [(u,v,d) for u,v,d in G.edges(data = True) if d['t']>=time_interval[0] 
          and d['t']<time_interval[0] + len_step]
    G1.add_edges_from(edges1)
    
    G2 = nx.Graph()
    edges2 = [(u,v,d) for u,v,d in G.edges(data = True) if d['t']>=time_interval[0] + len_step 
          and d['t']<time_interval[0]+2*len_step]
    G2.add_edges_from(edges2)
    
    G3 = nx.Graph()
    edges3 = [(u,v,d) for u,v,d in G.edges(data = True) if d['t']>=time_interval[0] + 2*len_step 
          and d['t']<time_interval[0]+3*len_step]
    G3.add_edges_from(edges3)
    
    G4 = nx.Graph()
    edges4 = [(u,v,d) for u,v,d in G.edges(data = True) if d['t']>=time_interval[0] + 3*len_step 
          and d['t']<time_interval[1]]
    G4.add_edges_from(edges4)
    
    return G1, G2, G3, G4





















