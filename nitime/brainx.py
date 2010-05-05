"""
brainx is a module that computes and represents quantities derived from graph
theory. These quantities can be useful in the analysis of large connectivity
matrices, for example those derived from coherency or correlation analysis.

This module makes use of networkx (http://networkx.lanl.gov/), which is a
library for analysis of networks. Therefore, this library is a dependence of
this module.

"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# From stdlib
import math
import random
import copy

#Third party 
import networkx as nx
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import time


#-----------------------------------------------------------------------------
# Metrics - compute various useful metrics of networks:
#-----------------------------------------------------------------------------
def nodal_pathlengths(G,n_nodes):
    """ Compute mean path length for each node.
    Note: it is unclear how to treat infinite path lengths.  For now, I replace them with np.inf, but this may make taking the mean later on difficult
    Inputs: G graph data output from mkgraph; n_nodes number of nodes in graph"""
    nodal_means=np.zeros((n_nodes),dtype=float)
    lengths= nx.all_pairs_shortest_path_length(G)
    for src,pair in lengths.iteritems():
        source_paths=[]
        source_arr=np.array([])
        for targ,val in pair.items():
            if src==targ:
                continue # we want to include src,target repeats, right?
            source_paths.append(float(val))
            source_arr=np.array(source_paths)
        if source_arr.size==0: #make the mean path length 0 if node is disconnected
            source_arr=np.array([np.nan])
        nodal_means[src]=source_arr.mean()
    #nodal_array=np.array(nodal_means)
    return nodal_means

def assert_no_selfloops(G):
    """Raise an error if the graph G has any selfloops.
    """
    if G.nodes_with_selfloops():
        raise ValueError("input graph can not have selfloops")

#@profile
def path_lengths(G):
    """Compute array of all shortest path lengths for the given graph.

    The length of the output array is the number of unique pairs of nodes that
    have a connecting path, so in general it is not known in advance.

    This assumes the graph is undirected, as for any pair of reachable nodes,
    once we've seen the pair we do not keep the path length value for the
    inverse path.
    
    Parameters
    ----------
    G : an undirected graph object.
    """

    assert_no_selfloops(G)
    
    length = nx.all_pairs_shortest_path_length(G)
    paths = []
    seen = set()
    for src,targets in length.iteritems():
        seen.add(src)
        neigh = set(targets.keys()) - seen
        paths.extend(targets[targ] for targ in neigh)
    
    
    return np.array(paths) 


#@profile
def path_lengthsSPARSE(G):
    """Compute array of all shortest path lengths for the given graph.

    XXX - implementation using scipy.sparse.  This might be faster for very
    sparse graphs, but so far for our cases the overhead of handling the sparse
    matrices doesn't seem to be worth it.  We're leaving it in for now, in case
    we revisit this later and it proves useful.

    The length of the output array is the number of unique pairs of nodes that
    have a connecting path, so in general it is not known in advance.

    This assumes the graph is undirected, as for any pair of reachable nodes,
    once we've seen the pair we do not keep the path length value for the
    inverse path.
    
    Parameters
    ----------
    G : an undirected graph object.
    """

    assert_no_selfloops(G)
    
    length = nx.all_pairs_shortest_path_length(G)

    nnod = G.number_of_nodes()
    paths_mat = sparse.dok_matrix((nnod,nnod))
    
    for src,targets in length.iteritems():
        for targ,val in targets.items():
            paths_mat[src,targ] = val

    return sparse.triu(paths_mat,1).data


def glob_efficiency(G):
    """Compute array of global efficiency for the given graph.

    Global efficiency: returns a list of the inverse path length matrix
    across all nodes The mean of this value is equal to the global efficiency
    of the network."""
    
    return 1.0/path_lengths(G)
        
def nodal_efficiency(G):
    """Compute array of global efficiency for the given graph.

    Nodal efficiency: XXX - define."""
        
    nodepaths=[]
    length = nx.all_pairs_shortest_path_length(G)
    for src,targets in length.iteritems():
        paths=[]
        for targ,val in targets.items():
            if src==targ:
                continue
            
            paths.append(1.0/val)
        
        nodepaths.append(np.mean(paths))
        
    return np.array(nodepaths) 

#@profile
def local_efficiency(G):
    """Compute array of global efficiency for the given grap.h

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    nodepaths=[]
    length=nx.all_pairs_shortest_path_length(G)
    for n in G.nodes():
        nneighb= nx.neighbors(G,n)
        
        paths=[]
        for src,targets in length.iteritems():
            for targ,val in targets.iteritems():
                val=float(val)
                if src==targ:
                    continue
                if src in nneighb and targ in nneighb:
                    
                    paths.append(1/val)
        
        p=np.array(paths)
        psize=np.size(p)
        if (psize==0):
            p=np.array(0)
            
        nodepaths.append(p.mean())
    
    return np.array(nodepaths)


#@profile
def local_efficiency(G):
    """Compute array of local efficiency for the given graph.

    Local efficiency: returns a list of paths that represent the nodal
    efficiencies across all nodes with their direct neighbors"""

    assert_no_selfloops(G)

    nodepaths = []
    length = nx.all_pairs_shortest_path_length(G)
    for n in G:
        nneighb = set(nx.neighbors(G,n))

        paths = []
        for nei in nneighb:
            other_neighbors = nneighb - set([nei])
            nei_len = length[nei]
            paths.extend( [nei_len[o] for o in other_neighbors] )

        if paths:
            p = 1.0 / np.array(paths,float)
            nodepaths.append(p.mean())
        else:
            nodepaths.append(0.0)
                
    return np.array(nodepaths)


def dynamical_importance(G):
    """Compute dynamical importance for G.

    Ref: Restrepo, Ott, Hunt. Phys. Rev. Lett. 97, 094102 (2006)
    """
    # spectrum of the original graph
    eigvals = nx.adjacency_spectrum(G)
    lambda0 = eigvals[0]
    # Now, loop over all nodes in G, and for each, make a copy of G, remove
    # that node, and compute the change in lambda.
    nnod = G.number_of_nodes()
    dyimp = np.empty(nnod,float)
    for n in range(nnod):
        gn = G.copy()
        gn.remove_node(n)
        lambda_n = nx.adjacency_spectrum(gn)[0]
        dyimp[n] = lambda0 - lambda_n
    # Final normalization
    dyimp /= lambda0
    return dyimp


def weighted_degree(G):
    """Return an array of degrees that takes weights into account.

    For unweighted graphs, this is the same as the normal degree() method
    (though we return an array instead of a list).
    """
    amat = nx.adj_matrix(G).A  # get a normal array out of it
    return abs(amat).sum(0)  # weights are sums across rows


def graph_summary(G):
    """Compute a set of statistics summarizing the structure of a graph.
    
    Parameters
    ----------
    G : a graph object.

    threshold : float, optional

    Returns
    -------
      Mean values for: lp, clust, glob_eff, loc_eff, in a dict.
    """
    
    # Average path length
    lp = path_lengths(G)
    clust = np.array(nx.clustering(G))
    glob_eff = glob_efficiency(G)
    loc_eff = local_efficiency(G)
    
    return dict( lp=lp.mean(), clust=clust.mean(), glob_eff=glob_eff.mean(),
                 loc_eff=loc_eff.mean() )

def nodal_summaryOut(G, n_nodes):
    """ Compute statistics for individual nodes

    Parameters
    ----------
    G: graph data output from mkgraph
    out: array output from nodal_summaryOut, so can keep appending
    cost: cost value for these calculations
    n_nodes: number of nodes in graph.

    Returns
    -------

    A dict with: lp, clust, b_cen, c_cen, nod_eff, loc_eff, degree."""

    lp = nodal_pathlengths(G,n_nodes) #can't use the regular one, because it substitutes [] for disconnected nodes
    clust = np.array(nx.clustering(G))
    b_cen = np.array(nx.betweenness_centrality(G).values())
    c_cen = np.array(nx.closeness_centrality(G).values())
    nod_eff=nodal_efficiency(G)
    loc_eff=local_efficiency(G)
    deg = G.degree()

    return dict(lp=lp, clust=clust, b_cen=b_cen, c_cen=c_cen, nod_eff=nod_eff,
                loc_eff=loc_eff,deg=deg)


#-----------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------
def store_metrics(b, s, co, metd, arr):
    """Store a set of metrics into a structured array"""

    if arr.ndim == 3:
        idx = b,s,co
    elif arr.ndim == 4:
        idx = b,s,co,slice(None)
    else:
        raise ValueError("only know how to handle 3 or 4-d arrays")
    
    for met_name, met_val in metd.iteritems():
        arr[idx][met_name] = met_val
    

def regular_lattice(n,k):
    """Return a regular lattice graph with n nodes and k neighbor connections.

    This graph consists of a ring with n nodes which then get connected to
    their k (k-1 if k is odd) nearest neighbors.

    This type of graph is the starting point for the Watts-Strogatz small-world
    model, where connections are then rewired in a second phase.
    """
    # Code simplified from the networkx.watts_strogatz_graph one
    G = nx.Graph()
    G.name="regular_lattice(%s,%s)"%(n,k)
    nodes = range(n) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k/2+1):
        targets = nodes[j:] + nodes[:j] # first j nodes are now last in list
        G.add_edges_from(zip(nodes,targets))
    return G


def compile_data(input,tmslabel,mat_type,scale,data_type):
    """This function reads in data into a text file"""
    filename='Mean_'+data_type+'_'+tmslabel+'_'+mat_type+scale+'.txt'
    f=open(filename,'a')
    for i in range(0,len(input)):
        f.write('%s\t' %input[i])
    f.write('\n')
    f.close()


def arr_stat(x,ddof=1):
    """Return (mean,stderr) for the input array"""
    m = x.mean()
    std = x.std(ddof=ddof)
    return m,std


def threshold_arr(cmat,threshold=0.0,threshold2=None):
    """Threshold values from the input matrix.

    Parameters
    ----------
    cmat : array
    
    threshold : float, optional.
      First threshold.
      
    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    indices, values: a tuple with ndim+1
    
    Examples
    --------
    >>> a = np.linspace(0,0.8,7)
    >>> a
    array([ 0.    ,  0.1333,  0.2667,  0.4   ,  0.5333,  0.6667,  0.8   ])
    >>> threshold_arr(a,0.3)
    (array([3, 4, 5, 6]), array([ 0.4   ,  0.5333,  0.6667,  0.8   ]))

    With two thresholds:
    >>> threshold_arr(a,0.3,0.6)
    (array([0, 1, 2, 5, 6]), array([ 0.    ,  0.1333,  0.2667,  0.6667,  0.8   ]))

    """
    # Select thresholds
    if threshold2 is None:
        th_low = -np.inf
        th_hi  = threshold
    else:
        th_low = threshold
        th_hi  = threshold2

    # Mask out the values we are actually going to use
    idx = np.where( (cmat < th_low) | (cmat > th_hi) )
    vals = cmat[idx]
    
    return idx + (vals,)


def thresholded_arr(arr,threshold=0.0,threshold2=None,fill_val=np.nan):
    """Threshold values from the input matrix and return a new matrix.

    Parameters
    ----------
    arr : array
    
    threshold : float
      First threshold.
      
    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    An array shaped like the input, with the values outside the threshold
    replaced with fill_val.
    
    Examples
    --------
    """
    a2 = np.empty_like(arr)
    a2.fill(fill_val)
    mth = threshold_arr(arr,threshold,threshold2)
    idx,vals = mth[:-1], mth[-1]
    a2[idx] = vals
    
    return a2


def normalize(arr,mode='direct',folding_edges=None):
    """Normalize an array to [0,1] range.

    By default, this simply rescales the input array to [0,1].  But it has a
    special 'folding' mode that allows for the normalization of an array with
    negative and positive values by mapping the negative values to their
    flipped sign

    Parameters
    ----------
    arr : 1d array
    
    mode : string, one of ['direct','folding']

    folding_edges : (float,float)
      Only needed for folding mode, ignored in 'direct' mode.

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> a = np.linspace(0.3,0.8,7)
    >>> normalize(a)
    array([ 0.    ,  0.1667,  0.3333,  0.5   ,  0.6667,  0.8333,  1.    ])
    >>> 
    >>> b = np.concatenate([np.linspace(-0.7,-0.3,4),
    ...                     np.linspace(0.3,0.8,4)] )
    >>> b
    array([-0.7   , -0.5667, -0.4333, -0.3   ,  0.3   ,  0.4667,  0.6333,  0.8   ])
    >>> normalize(b,'folding',[-0.3,0.3])
    array([ 0.8   ,  0.5333,  0.2667,  0.    ,  0.    ,  0.3333,  0.6667,  1.    ])
    >>> 
    >>> 
    >>> c = np.concatenate([np.linspace(-0.8,-0.3,4),
    ...                     np.linspace(0.3,0.7,4)] )
    >>> c
    array([-0.8   , -0.6333, -0.4667, -0.3   ,  0.3   ,  0.4333,  0.5667,  0.7   ])
    >>> normalize(c,'folding',[-0.3,0.3])
    array([ 1.    ,  0.6667,  0.3333,  0.    ,  0.    ,  0.2667,  0.5333,  0.8   ])
    """
    if mode == 'direct':
        return rescale_arr(arr,0,1)
    else:
        fa, fb = folding_edges
        amin, amax = arr.min(), arr.max()
        ra,rb = float(fa-amin),float(amax-fb) # in case inputs are ints
        if ra<0 or rb<0:
            raise ValueError("folding edges must be within array range")
        greater = arr>= fb
        upper_idx = greater.nonzero()
        lower_idx = (~greater).nonzero()
        # Two folding scenarios, we map the thresholds to zero but the upper
        # ranges must retain comparability.
        if ra > rb:
            lower = 1.0 - rescale_arr(arr[lower_idx],0,1.0)
            upper = rescale_arr(arr[upper_idx],0,float(rb)/ra)
        else:
            upper = rescale_arr(arr[upper_idx],0,1)
            # The lower range is trickier: we need to rescale it and then flip
            # it, so the edge goes to 0.
            resc_a = float(ra)/rb
            lower = rescale_arr(arr[lower_idx],0,resc_a)
            lower = resc_a - lower
        # Now, make output array
        out = np.empty_like(arr)
        out[lower_idx] = lower
        out[upper_idx] = upper
        return out


def mat2graph(cmat,threshold=0.0,threshold2=None):
    """Make a weighted graph object out of an adjacency matrix.

    The values in the original matrix cmat can be thresholded out.  If only one
    threshold is given, all values below that are omitted when creating edges.
    If two thresholds are given, then values in the th2-th1 range are
    ommitted.  This allows for the easy creation of weighted graphs with
    positive and negative values where a range of weights around 0 is omitted.
    
    Parameters
    ----------
    cmat : 2-d square array
      Adjacency matrix.
    threshold : float
      First threshold.
    threshold2 : float
      Second threshold.

    Returns
    -------
    G : a NetworkX weighted graph object, to which a dictionary called
    G.metadata is appended.  This dict contains the original adjacency matrix
    cmat, the two thresholds, and the weights 
    """ 

    # Input sanity check
    nrow,ncol = cmat.shape
    if nrow != ncol:
        raise ValueError("Adjacency matrix must be square")

    row_idx, col_idx, vals = threshold_arr(cmat,threshold,threshold2)
    # Also make the full thresholded array available in the metadata
    cmat_th = np.empty_like(cmat)
    if threshold2 is None:
        cmat_th.fill(threshold)
    else:
        cmat_th.fill(-np.inf)
    cmat_th[row_idx,col_idx] = vals

    # Next, make a normalized copy of the values.  For the 2-threshold case, we
    # use 'folding' normalization
    if threshold2 is None:
        vals_norm = normalize(vals)
    else:
        vals_norm = normalize(vals,'folding',[threshold,threshold2])

    # Now make the actual graph
    G = nx.Graph(weighted=True)
    G.add_nodes_from(range(nrow))
    # To keep the weights of the graph to simple values, we store the
    # normalize ones in a separate dict that we'll stuff into the graph
    # metadata.
    
    normed_values = {}
    for i,j,val,nval in zip(row_idx,col_idx,vals,vals_norm):
        if i == j:
            # no self-loops
            continue
        G.add_edge(i,j,weight=val)
        normed_values[i,j] = nval

    # Write a metadata dict into the graph and save the threshold info there
    G.metadata = dict(threshold1=threshold,
                      threshold2=threshold2,
                      cmat_raw=cmat,
                      cmat_th =cmat_th,
                      vals_norm = normed_values,
                      )
    return G

# Backwards compatibility name
mkgraph = mat2graph

def mkdigraph(cmat,dmat,threshold=0.0,threshold2=None):
    """Make a graph object out of an adjacency matrix and direction matrix"""

    # Input sanity check
    nrow,ncol = cmat.shape
    if not nrow==ncol:
        raise ValueError("Adjacency matrix must be square")

    row_idx, col_idx, vals = threshold_arr(cmat,threshold,threshold2)

    # Now make the actual graph
    G = nx.DiGraph()
    G.add_nodes_from(range(nrow))

    for i,j,val in zip(row_idx,col_idx,vals):
        if dmat[i,j] > 0:
            G.add_edge(i,j,val)
        else:
            G.add_edge(j,i,val)

    return G


def rescale_arr(arr,amin,amax):
    """Rescale an array to a new range.

    Return a new array whose range of values is (amin,amax).

    Parameters
    ----------
    arr : array-like

    amin : float
      new minimum value

    amax : float
      new maximum value

    Examples
    --------
    >>> a = np.arange(5)

    >>> rescale_arr(a,3,6)
    array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])
    """
    
    # old bounds
    m = arr.min()
    M = arr.max()
    # scale/offset
    s = float(amax-amin)/(M-m)
    d = amin - s*m
    
    # Apply clip before returning to cut off possible overflows outside the
    # intended range due to roundoff error, so that we can absolutely guarantee
    # that on output, there are no values > amax or < amin.
    return np.clip(s*arr+d,amin,amax)


# backwards compatibility only, deprecated
def replace_diag(arr,val=0):
    fill_diagonal(arr,val)
    return arr


def cost2thresh(cost,sub,bl,lk,last):
    """A definition for loading the lookup table and finding the threshold associated with a particular cost for a particular subject in a particular block
    
    inputs:
    cost: cost value for which we need the associated threshold
    sub: subject number
    bl: block number
    lk: lookup table (block x subject x cost
    last: last threshold value

    output:
    th: threshold value for this cost"""

    #print cost,sub,bl
    
    ind=np.where(lk[bl][sub][1]==cost)
    th=lk[bl][sub][0][ind]

    if len(th)>1:
        th=th[0] #if there are multiple thresholds, go down to the lower cost ####Is this right?!!!####
        print 'multiple thresh'
    elif len(th)<1:
        th=last #if there is no associated thresh value because of repeats, just use the previous one
        print 'use previous thresh'
    else:
        th=th[0]
      
    #print th    
    return th


def network_ind(ntwk_type,n_nodes):
    """Reads in a network type, number of nodes total and returns the indices of that network"""

    net_core ="dACC L_aIfO R_aIfO L_aPFC R_aPFC L_aThal R_aThal".split()
    net_fp = """L_frontcx  R_frontcx    L_IPL  R_IPL    L_IPS  R_IPS  L_PFC
    R_PFC L_precuneus    R_precuneus  midcing""".split()
    net_motor = """L_motor  R_motor L_preSMA R_preSMA SMA""".split()
    net_aal = " "
    
    subnets = { 'g': net_core,
                'b': net_fp,
                'y': net_motor,
                }
    ALL_LABELS = net_core+net_fp +net_motor

    if ntwk_type=='core':
        roi_ind=range(0,7)
        subnets = { 'g': net_core}
        ALL_LABELS = net_core
    elif ntwk_type=='FP':
        roi_ind=range(7,18)
        subnets = {'b': net_fp}
        ALL_LABELS = net_fp
    elif ntwk_type=='all':
        roi_ind=range(0,n_nodes)
        subnets = { 'g': net_core,
            'b': net_fp }#,
            #'y': net_motor,
            #}
        ALL_LABELS = net_core+net_fp# +net_motor
    elif ntwk_type=='aal':
        roi_ind=range(0,n_nodes)
        subnets = {'k': net_aal}
        ALL_LABELS = net_aal
    else:
        print 'do not recognize network type'
    return roi_ind,subnets,ALL_LABELS


#-----------------------------------------------------------------------------
# Modularity
#-----------------------------------------------------------------------------

"""Detect modules in a network.

Citation: He Y, Wang J, Wang L, Chen ZJ, Yan C, et al. (2009) Uncovering
Intrinsic Modular Organization of Spontaneous Brain Activity in Humans. PLoS
ONE 4(4): e5226. doi:10.1371/journal.pone.0005226

Comparing community structure identification
J. Stat. Mech. (2005) P0900
Leon Danon1,2, Albert Diaz-Guilera1, Jordi Duch2 and  Alex Arenas
Online at stacks.iop.org/JSTAT/2005/P09008
doi:10.1088/1742-5468/2005/09/P09008

"""

class GraphPartition(object):
    """Represent a graph partition."""

    def __init__(self, graph, index):
        """New partition, given a graph and a dict of module->nodes.

        Parameters
        ----------
        graph : network graph instance
          Graph to which the partition index refers to.
          
        index : dict
          A dict that maps module labels to sets of nodes, this describes the
          partition in full.

        Note
        ----
        The values in the index dict MUST be real sets, not lists.  No checks
        are made of this fact, but later the code relies on them being sets and
        may break in strange manners if the values were stored in non-set
        objects.
        """
        # Store references to the original graph and label dict
        self.index = copy.deepcopy(index)
        #self.graph = graph
        
        # We'll need the graph's adjacency matrix often, so store it once
        self.graph_adj_matrix = nx.adj_matrix(graph)
        
        # Just to be sure, we don't want to count self-links, so we zero out the
        # diagonal.
        util.fill_diagonal(self.graph_adj_matrix, 0)

        # Store statically a few things about the graph that don't change (as
        # long as the graph does not change
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = graph.number_of_edges()

        # Store the nodes as a set, needed for many operations
        self._node_set = set(graph.nodes())

        # Now, build the edge information used in modularity computations
        self.mod_e, self.mod_a = self._edge_info()
    

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.index)

    def _edge_info(self, mod_e=None, mod_a=None, index=None):
        """Create the vectors of edge information.
        
        Returns
        -------
          mod_e: diagonal of the edge matrix E

          mod_a: sum of the rows of the E matrix
        """
        num_mod = len(self)
        if mod_e is None: mod_e = [0] * num_mod
        if mod_a is None: mod_a = [0] * num_mod
        if index is None: index = self.index
        
        norm_factor = 1.0/(2.0*self.num_edges)
        mat = self.graph_adj_matrix
        set_nodes = self._node_set
        for m,modnodes in index.iteritems():
            #set_modnodes=set(modnodes)
            #btwnnodes   = list(set_nodes - modnodes)
            btwnnodes = list(set_nodes - set(index[m]))
            modnodes  = list(modnodes)
            #why isnt' self.index a set already?  graph_partition.index[m]
            #looks like a set when we read it in ipython
            mat_within  = mat[modnodes,:][:,modnodes]
            mat_between = mat[modnodes,:][:,btwnnodes]
            perc_within = mat_within.sum() * norm_factor
            perc_btwn   = mat_between.sum() * norm_factor
            mod_e[m] = perc_within #all of the E's
            mod_a[m] = perc_btwn+perc_within #all of the A's
            #mod_e.append(perc_within)
            #mod_a.append(perc_btwn+perc_within)

            
        return mod_e, mod_a

    def modularity_newman(self):
        """ Function using other version of expressing modularity, from the Newman papers (2004 Physical Review)

        Parameters:
        g = graph
        part = partition

        Returns:
        mod = modularity
        """
        return (np.array(self.mod_e) - (np.array(self.mod_a)**2)).sum()

    modularity = modularity_newman
    #modularity = modularity_guimera


    ## def modularity_guimera(self, g, part):
    ##     """This function takes in a graph and a partition and returns Newman's
    ##     modularity for that graph"""

    ##     """ Parameters
    ##     # g = graph part = partition; a dictionary that contains a list of
    ##     # nodes that make up that module"""

    ##     #graph values
    ##     num_mod = len(part)
    ##     L = nx.number_of_edges(g)
    ##     # construct an adjacency matrix from the input graph (g)
    ##     mat = nx.adj_matrix(g)

    ##     M = 0
    ##     # loop over the modules in the graph, create an adjacency matrix
    ##     for m, val in part.iteritems():
    ##         #create a 'sub mat'
    ##         submat = mat[val,:][:,val]

    ##         #make a graph
    ##         subg = nx.from_numpy_matrix(submat)

    ##         #calculate module-specific metrics
    ##         link_s = float(subg.number_of_edges())
    ##         deg_s = np.sum(nx.degree(g,val), dtype=float)

    ##         #compute modularity!
    ##         M += ((link_s/L) - (deg_s/(2*L))**2)

    ##     return M
    
    def compute_module_merge(self, m1, m2):
        """Merges two modules in a given partition.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
          
        Returns
        -------

          """
        # Below, we want to know that m1<m2, so we enforce that:
        if m1>m2:
            m1, m2 = m2, m1

        # Pull from m2 the nodes and merge them into m1
        merged_module = self.index[m1] | self.index[m2]
        
        #make an empty matrix for computing "modularity" level values
        e1 = [0]
        a1 = [0]
        e0, a0 = self.mod_e, self.mod_a
        
        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, {0:merged_module})
        
        # Compute the change in modularity
        delta_q =  (e1[0]-a1[0]**2) - \
            ( (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2) )

        #print 'NEW: ',e1,a1,e0[m1],a0[m1],e0[m2],a0[m2]
  
        return merged_module, e1[0], a1[0], -delta_q, 'merge',m1,m2,m2

    
    def apply_module_merge(self, m1, m2, merged_module, e_new, a_new):
        """Merges two modules in a given partition.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        XXX
        
        Returns
        -------
        XXX
          """

        # Below, we want to know that m1<m2, so we enforce that:
        if m1>m2:
            m1, m2 = m2, m1

        # Pull from m2 the nodes and merge them into m1
        self.index[m1] = merged_module
        del self.index[m2]

        # We need to shift the keys to account for the fact that we popped out
        # m2
        
        rename_keys(self.index,m2)
        
        self.mod_e[m1] = e_new
        self.mod_a[m1] = a_new
        self.mod_e.pop(m2)
        self.mod_a.pop(m2)
        
        
    def compute_module_split(self, m, n1, n2):
        """Splits a module into two new ones.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        m : module identifier
        n1, n2 : sets of nodes
          The two sets of nodes in which the nodes originally in module m will
          be split.  Note: It is the responsibility of the caller to ensure
          that the set n1+n2 is the full set of nodes originally in module m.
          
        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        # create a dict that contains the new modules 0 and 1 that have the sets n1 and n2 of nodes from module m.
        split_modules = {0: n1, 1: n2} 

        #make an empty matrix for computing "modularity" level values
        e1 = [0,0]
        a1 = [0,0]
        e0, a0 = self.mod_e, self.mod_a

        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, split_modules)

        # Compute the change in modularity
        delta_q =  ( (e1[0]-a1[0]**2) + (e1[1]- a1[1]**2) ) - \
            (e0[m]-a0[m]**2)
        
        return split_modules, e1, a1, -delta_q,'split',m,n1,n2

    
    def apply_module_split(self, m, n1, n2, split_modules, e_new, a_new):
        """Splits a module into two new ones.

        This updates in place the mod_e and mod_a arrays (both of which lose a
        row).  The new, merged module will be identified as m1.
        
        Parameters
        ----------
        m : module identifier
        n1, n2 : sets of nodes
          The two sets of nodes in which the nodes originally in module m will
          be split.  Note: It is the responsibility of the caller to ensure
          that the set n1+n2 is the full set of nodes originally in module m.
          
        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        # To reuse slicing code, use m1/m2 lables like in merge code
        m1 = m
        m2 = len(self)
        
        #Add a new module to the end of the index dictionary
        self.index[m1] = split_modules[0] #replace m1 with n1
        self.index[m2] = split_modules[1] #add in new module, fill with n2
        
        self.mod_e[m1] = e_new[0]
        self.mod_a[m1] = a_new[0]
        self.mod_e.insert(m2,e_new[1])
        self.mod_a.insert(m2,a_new[1])
        
        #self.mod_e[m2] = e_new[1]
        #self.mod_a[m2] = a_new[1]
        
        
    def node_update(self, n, m1, m2):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        #Update the index with the change
        index = self.index
        index[m1].remove(n)
        index[m2].add(n)

        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m1])<1:
            self.index.pop(m1)
            rename_keys(self.index,m1)
            
        # Before we overwrite the mod vectors, compute the contribution to
        # modularity from before the change
        e0, a0 = self.mod_e, self.mod_a
        mod_old = (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2)
        # Update in place mod vectors with new index
        self._edge_info(self.mod_e, self.mod_a, {m1:index[m1], m2:index[m2]})
        e1, a1 = self.mod_e, self.mod_a
        #Compute the change in modularity
        return (e1[m1]-a1[m1]**2) + (e1[m2]-a1[m2]**2) - mod_old

    def compute_node_update(self, n, m1, m2):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""
        
        n1 = self.index[m1]
        n2 = self.index[m2]

        node_moved_mods = {0: n1 - set([n]),1: n2 | set([n])}
            
        # Before we overwrite the mod vectors, compute the contribution to
        # modularity from before the change
        e1 = [0,0]
        a1 = [0,0]
        e0, a0 = self.mod_e, self.mod_a

        # The values that change: _edge_info with arguments will update the e,
        # a vectors only for the modules in index
        e1, a1 = self._edge_info(e1, a1, node_moved_mods)
        
        #Compute the change in modularity
        delta_q =  ( (e1[0]-a1[0]**2) + (e1[1]-a1[1]**2)) - \
            ( (e0[m1]-a0[m1]**2) + (e0[m2]-a0[m2]**2) )
        
        #print n,m1,m2,node_moved_mods,n1,n2
        return node_moved_mods, e1, a1, -delta_q, n, m1, m2

    def apply_node_update(self, n, m1, m2, node_moved_mods, e_new, a_new):
        """Moves a single node within or between modules

        Parameters
        ----------
        n : node identifier
          The node that will be moved from module m1 to module m2
        m1 : module identifier
          The module that n used to belong to.
        m2 : module identifier
          The module that n will now belong to.

        Returns
        -------
          The change in modularity resulting from the change
          (Q_final-Q_initial)"""

        
        
        self.index[m1] = node_moved_mods[0]
        self.index[m2] = node_moved_mods[1]
        
        # This checks whether there is an empty module. If so, renames the keys.
        if len(self.index[m1])<1:
            self.index.pop(m1)
            rename_keys(self.index,m1)
            
        self.mod_e[m1] = e_new[0]
        self.mod_a[m1] = a_new[0]
        self.mod_e[m2] = e_new[1]
        self.mod_a[m2] = a_new[1]

    def random_mod(self):
        """Makes a choice whether to merge or split modules in a partition
        
        Returns:
        -------
        if splitting: m1, n1, n2
          m1: the module to split
          n1: the set of nodes to put in the first output module
          n2: the set of nodes to put in the second output module

        if merging: m1, m2
          m1: module 1 to merge
          m2: module 2 to merge
        """

        # number of modules in the partition
        num_mods=len(self)
        
        
        # Make a random choice bounded between 0 and 1, less than 0.5 means we will split the modules
        # greater than 0.5 means we will merge the modules.
        
        if num_mods >= self.num_nodes-1:
            coin_flip = 1 #always merge if each node is in a separate module
        elif num_mods <= 2:
            coin_flip = 0 #always split if there's only one module
        else:
            coin_flip = random.random()
            

        #randomly select two modules to operate on
        rand_mods = np.random.permutation(range(num_mods))
        m1 = rand_mods[0]
        m2 = rand_mods[1]

        if coin_flip > 0.5:
            #merge
            #return self.module_merge(m1,m2)
            return self.compute_module_merge(m1,m2)
        else: 
            #split
            # cannot have a module with less than 1 node
            while len(self.index[m1]) <= 1:

                #reselect the first  module
                rand_mods = np.random.permutation(range(num_mods))
                m1 = rand_mods[0]
                #m1 = random.randint(0,num_mods)

            # list of nodes within that module
            list_nods = list(self.index[m1])

            # randomly partition the list of nodes into 2
            nod_split_ind = random.randint(1,len(list_nods)) #can't pick the first node as the division
            n1 = set(list_nods[:nod_split_ind])
            n2 = set(list_nods[nod_split_ind:])

            #We may want to return output of merging/splitting directly, but
            #for now we're returning inputs for those modules.
            
            return self.compute_module_split(m1,n1,n2)


    def random_mod_old(self):
        """Makes a choice whether to merge or split modules in a partition
        
        Returns:
        -------
        if splitting: m1, n1, n2
          m1: the module to split
          n1: the set of nodes to put in the first output module
          n2: the set of nodes to put in the second output module

        if merging: m1, m2
          m1: module 1 to merge
          m2: module 2 to merge
        """

        # number of modules in the partition
        num_mods=len(self)
        
        
        # Make a random choice bounded between 0 and 1, less than 0.5 means we will split the modules
        # greater than 0.5 means we will merge the modules.
        
        if num_mods >= self.num_nodes-1:
            coin_flip = 1 #always merge if each node is in a separate module
        elif num_mods <= 2:
            coin_flip = 0 #always split if there's only one module
        else:
            coin_flip = random.random()
            
        #randomly select two modules to operate on
        rand_mods = np.random.permutation(range(num_mods))
        m1 = rand_mods[0]
        m2 = rand_mods[1]

        if coin_flip > 0.5:
            #merge
            #return self.module_merge(m1,m2)
            return self.module_merge(m1,m2)
        else: 
            #split
            # cannot have a module with less than 1 node
            while len(self.index[m1]) <= 1:

                #reselect the first  module
                rand_mods = np.random.permutation(range(num_mods))
                m1 = rand_mods[0]
                #m1 = random.randint(0,num_mods)

            # list of nodes within that module
            list_nods = list(self.index[m1])

            # randomly partition the list of nodes into 2
            nod_split_ind = random.randint(1,len(list_nods)) #can't pick the first node as the division
            n1 = set(list_nods[:nod_split_ind])
            n2 = set(list_nods[nod_split_ind:])

            #We may want to return output of merging/splitting directly, but
            #for now we're returning inputs for those modules.
            
            return self.module_split(m1,n1,n2)
       
    def random_node(self):
        """ Randomly reassign one node from one module to another

        Returns:
        -------

        n: node to move
        m1: module node is currently in
        m2: module node will be moved to """

        # number of modules in the partition
        num_mods=len(self)
        if num_mods < 2:
            raise ValueError("Can not reassign node with only one module")

        # initialize a variable so we can search the modules to find one with
        # at least 1 node
        node_len = 0
        
        # select 2 random modules (the first must have at least 2 nodes in it)
        while node_len <= 1:
            
            # randomized list of modules
            rand_mods=np.random.permutation(range(num_mods))
            
            node_len = len(self.index[rand_mods[0]])
        
        m1 = rand_mods[0]
        m2 = rand_mods[1]

            
        # select a random node within one module
        node_list = list(self.index[m1])
        rand_perm = np.random.permutation(node_list)
        n = rand_perm[0]
        
        return self.compute_node_update(n,m1,m2)
    
#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def diag_stack(tup):
    """Stack arrays in sequence diagonally (block wise).
    
    Take a sequence of arrays and stack them diagonally to make a single block
    array.
    
    
    Parameters
    ----------
    tup : sequence of ndarrays
        Tuple containing arrays to be stacked. The arrays must have the same
        shape along all but the first two axes.
    
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.
    
    See Also
    --------
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    concatenate : Join a sequence of arrays together.
    vsplit : Split array into a list of multiple sub-arrays vertically.
    
    
    Examples
    --------
    """
    # Find number of rows and columns needed
    shapes = np.array([a.shape for a in tup], int)
    sums = shapes.sum(0)
    nrow = sums[0]
    ncol = sums[1]
    out = np.zeros((nrow, ncol), tup[0].dtype)
    row_offset = 0
    col_offset = 0
    for arr in tup:
        nr, nc = arr.shape
        row_end = row_offset+nr
        col_end = col_offset+nc
        out[row_offset:row_end, col_offset:col_end] = arr
        row_offset, col_offset = row_end, col_end
    return out


def random_modular_graph(nnod, nmod, av_degree, between_fraction=0.0):
    """
    Parameters
    ----------

    nnod : int
      Total number of nodes in the graph.

    nmod : int
      Number of modules.  Note that nmod must divide nnod evenly.

    av_degree : int
      Average degree of the nodes.

    between_fraction : float
      A number in [0,1], indicating the fraction of edges in each module which
      are wired to go between modules.
    """
    # sanity checks:
    if nnod%nmod:
        raise ValueError("nmod must divide nnod evenly")

    # Compute the number of nodes per module
    nnod_mod = nnod/nmod

    # The average degree requested can't be more than what the graph can
    # support if it were to be fully dense
    if av_degree > nnod_mod - 1:
        e = "av_degree can not be larger than (nnod_mod-1) = %i" % (nnod_mod-1)
        raise ValueError(e)

    # Compute the probabilities to generate the graph with, both for
    # within-module (p_in) and between-modules (p_out):
    z_out = between_fraction*av_degree
    p_in = (av_degree-z_out)/(nnod_mod-1.0)
    p_out = float(z_out)/(nnod-nnod_mod)

    # Some sanity checks
    assert 0 <= p_in <=1, "Invalid p_in=%s, not in [0,1]" % p_in
    assert 0 <= p_out <=1, "Invalid p_out=%s, not in [0,1]" % p_out

    # Create initial matrix with uniform random numbers in the 0-1 interval.
    mat = util.symm_rand_arr(nnod)

    # Create the masking matrix
    blocks = [np.ones((nnod_mod, nnod_mod))] * nmod
    mask = diag_stack(blocks)

    # Threshold the random matrix to create an actual adjacency graph.

    # Emi's trick: we need to use thresholding in only certain parts of the
    # matrix, corresponding to where the mask is 0 or 1.  Rather than having a
    # complex indexing operation, we'll just multiply the numbers in one region
    # by -1, and then we can do the thresholding over negative and positive
    # values. As long as we correct for this, it's a much simpler approach.
    mat[mask==1] *= -1

    adj = np.zeros((nnod, nnod))
    # Careful to flip the sign of the thresholding for p_in, since we used the
    # -1 trick above
    adj[np.logical_and(0 >= mat, mat > -p_in)] = 1
    adj[np.logical_and(0 < mat, mat < p_out)] = 1

    # no self-links
    util.fill_diagonal(adj, 0)
    # Our return object is a graph, not the adjacency matrix
    return nx.from_numpy_matrix(adj)


def array_to_string(part):
    """The purpose of this function is to convert an array of numbers into
    a list of strings. Mainly for use with the plot_partition function that
    requires a dict of strings for node labels.

    """

    out_part=dict.fromkeys(part)
    
    for m in part.iterkeys():
        out_part[m]=str(part[m])
    
    return out_part


def rename_keys(dct, key):
    """This function reads in a partition and a single module to be
    removed,pops out the value(s) and shifts the key names accordingly.

    Parameters
    ----------
    XXX
    
    Returns
    -------
    XXX
    """
 
    for m in range(key,len(dct)):
        dct[m] = dct.pop(m+1)

def rand_partition(g):
    """This function takes in a graph and returns a dictionary of labels for
    each node. Eventually it needs to be part of the simulated annealing program,
    but for now it will just make a random partition."""

    num_nodes = g.number_of_nodes()

    # randomly select a number of modules
    num_mods = random.randint(1,num_nodes)

    # randomize the order of nodes into a list
    rand_nodes = np.random.permutation(num_nodes)

    # We'll use this twice below, don't re-generate it.
    mod_range = range(num_mods)
    
    # set up a dictionary containing each module and the nodes under it.
    # Note: the following loop *does* cover the entire range, even if it
    # doesn't appear obvious immediately.  The easiest way to see this is to
    # write the execution of the loop row-wise, assuming an ordered permutation
    # (rand_nodes), and then to read it column-wise.  It will be then obvious
    # that when each column ends at the last row, the next column starts with
    # the next node in the list, and no node is ever skipped.
    out = [set(rand_nodes[i::num_mods]) for i in mod_range]

##     # a simpler version of the partitioning

##     # We need to split the list of nodes into (num_mods) partitions which means we need (num_mods-1) slices.
##     # The slices need to be in increasing order so we can use them as indices
##     rand_slices=sort(np.random.permutation(rand_nodes)[:num_mods-1])

##     # initialize a dictionary
##     out = dict()
##     # initialize the first element of the node list
##     init_node=0
##     for m in range_mods:
        
##         #length of the current module
##         len_mod=rand_slices[s]-init_node
##         out[mod_ind] = rand_nodes[init_node:len_mod+init_node]
##         init_node=rand_slices[m]
        
    # The output is the final partition
    return dict(zip(mod_range,out))


def perfect_partition(nmod,nnod_mod):
    """This function takes in the number of modules and number of nodes per module
    and returns the perfect partition depending on the number of modules
    where the module number is fixed according to random_modular_graph()"""
    
    #empty dictionary to fill with the correct partition
    part=dict()
    #set up a dictionary containing each module and the nodes under it
    for m in range(nmod):
        part[m]=set(np.arange(nnod_mod)+m*nnod_mod) #dict([(nmod,nnod)])# for x in range(num_mods)])
        #print 'Part ' + str(m) + ': '+ str(part[m])
    
    return part

def plot_partition(g,part,title,fname='figure',nod_labels = None, pos = None):
    """This function takes in a graph and a partition and makes a figure that
    has each node labeled according to its partition assignment"""

    fig=plt.figure()
    nnod = g.number_of_nodes()

    if nod_labels == None:
        nod_labels = dict(zip(range(nnod),range(nnod)))
    else:
        nod_labels = dict(zip(range(nnod),nod_labels))

    nod_labels = array_to_string(nod_labels)

    
    if pos == None:
        pos=nx.circular_layout(g)
    
    col=colors.cnames.keys()
    
    niter = 0
    for m,val in part.iteritems():
        nx.draw_networkx_nodes(g,pos,nodelist=list(val),node_color=col[niter],node_size=50)
        niter += 1

    
    nx.draw_networkx_labels(g,pos,nod_labels,font_size=6)    
    nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g))

    plt.title(title)
    plt.savefig(fname)
    plt.close()
    #plt.show()

def compare_dicts(d1,d2):
    """Function that reads in two dictionaries of sets (i.e. a graph partition) and assess how similar they are.
    Needs to be updated so that it can adjust this measure to include partitions that are pretty close."""
    

    if len(d1)>len(d2):
        longest_dict=len(d1)
    else:
        longest_dict=len(d2)
    check=0
    #loop through the keys in the first dict
    for m1,val1 in d1.iteritems():
        #compare to the values in each key of the second dict
        for m2,val2 in d2.iteritems():
            if val1 == val2:
                check+=1
    return float(check)/longest_dict
        

def mutual_information(d1,d2):
    """Function that reads in two dictionaries of sets (i.e. a graph partition) and assess how similar they are using mutual information as in Danon, Diaz-Guilera, Duch & Arenas, J Statistical Mechanics 2005.
    
    Inputs:
    ------
    d1 = dictionary of 'real communities'
    d2 = dictionary of 'found communities'
    """
    
    dlist = [d1,d2]
    #first get rid of any empty values and relabel the keys accordingly
    new_d2 = dict()
    old_d2=d2
    for d in dlist:
        items = d.items()
        sort_by_length = [[len(v[1]),v[0]] for v in items]
        sort_by_length.sort()
        
        counter=0
        for i in range(len(sort_by_length)):
            #if the module is not empty...
            if sort_by_length[i][0]>0:
                new_d2[counter]=d[sort_by_length[i][1]]
                counter+=1

        d2=new_d2
    
    #define a 'confusion matrix' where rows = 'real communities' and columns = 'found communities'
    #The element of N (Nij) = the number of nodes in the real community i that appear in the found community j
    rows = len(d1)
    cols = len(d2)
    N = np.empty((rows,cols))
    rcol = range(cols)
    for i in range(rows):
        for j in rcol:
            N[i,j] = len(d1[i] & d2[j])
         

    nsum_row = N.sum(0)[np.newaxis, :]
    nsum_col = N.sum(1)[:, np.newaxis]
    nn = nsum_row.sum()
    log = np.log
    nansum = np.nansum
    
    num = nansum(N*log(N*nn/(nsum_row*nsum_col)))
    den = nansum(nsum_row*log(nsum_row/nn)) + nansum(nsum_col*log(nsum_col/nn))

    return -2*num/den
        
def decide_if_keeping(dE,temperature):
    """Function which uses the rule from Guimera & Amaral (2005) Nature paper to decide whether or not to keep new partition

    Parameters:
    dE = delta energy 
    temperature = current state of the system
=
    Returns:
    keep = 1 or 0 to decide if keeping new partition """

    if dE <= 0:
        return True
    else:
        return random.random() < math.exp(-dE/temperature)

    
def simulated_annealing(g,temperature = 50, temp_scaling = 0.995, tmin=1e-5,
                        bad_accept_mod_ratio_max = 0.8 ,
                        bad_accept_nod_ratio_max = 0.8, accept_mod_ratio_min =
                        0.05, accept_nod_ratio_min = 0.05,
                        extra_info = False):

    """ This function does simulated annealing on a graph

    Parameters:
    g = graph #to anneal over
    temperature = 5777 #temperature of the sun in Kelvin, where we're starting
    tmin = 0.0 # minimum temperature
    n_nochanges = 25 # number of times to allow no change in modularity before
    breaking out of loop search

    Return:
    part = final partition
    M = final modularity """

    #Make a random partition for the graph
    nnod = g.number_of_nodes()
    nnod2 = nnod**2
    part = dict()
    #check if there is only one module or nnod modules
    while (len(part) <= 1) or (len(part) == nnod): 
        part = rand_partition(g)
    

    # make a graph partition object
    graph_partition = GraphPartition(g,part)
    
    # The number of times we switch nodes in a partition and the number of
    # times we modify the partition, at each temperature.  These values were
    # suggested by Guimera and Amaral, Nature 443, p895.  This is achieved
    # simply by running two nested loops of length nnod
    
    nnod = graph_partition.num_nodes
    rnod = range(nnod)

    #initialize some counters
    count = 0
    
    #Initialize empty lists for keeping track of values
    energy_array = []#negative modularity
    rej_array = []
    temp_array = []
    energy_best = 0
    
    energy = -graph_partition.modularity()
    energy_array.append(energy)

    while temperature > tmin:
        # Initialize counters
        bad_accept_mod = 0
        accept_mod = 0
        reject_mod = 0
        count_mod = 0
        count_bad_mod = 0.0001  # small offset to avoid occasional 1/0 errors
        
        for i_mod in rnod:
            # counters for module change attempts
            count_mod+=1
            count+=1
            
            # Assess energy change of a new partition without changing the partition
            calc_dict,e_new,a_new,delta_energy,movetype,p1,p2,p3 = graph_partition.random_mod()

            # Increase the 'count_bad_mod' if the new partition increases the energy
            if delta_energy > 0:
                count_bad_mod += 1

            # Decide whether the new partition is better than the old
            keep = decide_if_keeping(delta_energy,temperature)

            # Append the current temperature to the temp list
            temp_array.append(temperature)
            
            if keep:
                # this applies changes in place if energy decreased; the
                # modules will either be merged or split depending on a random
                # coin flip
                if movetype=='merge':
                    graph_partition.apply_module_merge(p1,p2,calc_dict,e_new,a_new)
                else:
                    graph_partition.apply_module_split(p1,p2,p3,calc_dict,e_new,a_new)
                
                # add the change in energy to the total energy
                energy += delta_energy
                accept_mod += 1 #counts times accept mod because lower energy
                
                # Increase the 'bad_accept_mod' if the new partition increases
                # the energy and was accepted
                if delta_energy > 0 :
                    bad_accept_mod += 1
            #else:
                #make a new graph partition with the last partition
                #reject_mod += 1
                #graph_partition = GraphPartition(g,graph_partition.index)
            if energy < energy_best:
                
                energy_best = energy
                
            energy_array.append(energy)   
            
            #break out if we are accepting too many "bad" options (early on)
            #break out if we are accepting too few options (later on)
            if count_mod > 10:
                bad_accept_mod_ratio =  float(bad_accept_mod)/(count_bad_mod)
                accept_mod_ratio = float(accept_mod)/(count_mod)
                #print 'ba_mod_r', bad_accept_mod_ratio  # dbg
                if (bad_accept_mod_ratio > bad_accept_mod_ratio_max) \
                        or (accept_mod_ratio < accept_mod_ratio_min):
                    #print 'MOD BREAK'
                    break

            bad_accept_nod = 0
            accept_nod = 0
            count_nod = 0
            count_bad_nod =  0.0001 # init at 1 to avoid 1/0 errors later
            
            for i_nod in rnod:
                count_nod+=1
                count+=1

                #if (np.mod(count,10000)==0) and (temperature < 1e-1):
                #    plot_partition(g,part,'../SA_graphs2/try'+str(count)+'.png')

                # Assess energy change of a new partition
                calc_dict,e_new,a_new,delta_energy,p1,p2,p3 = graph_partition.random_node()
                if delta_energy > 0:
                    count_bad_nod += 1
                temp_array.append(temperature)
                
                keep = decide_if_keeping(delta_energy,temperature)

                if keep:
                    
                    graph_partition.apply_node_update(p1,p2,p3,calc_dict,e_new,a_new)
                    energy += delta_energy
                    accept_nod += 1
                    if delta_energy > 0 :
                        bad_accept_nod += 1
                #else:
                    #graph_partition = GraphPartition(g,graph_partition.index)
                if energy < energy_best:
                    energy_best = energy
                    
                energy_array.append(energy)
                
                #break out if we are accepting too many "bad" options (early on)
                #break out if we are accepting too few options (later on)
                if count_nod > 10:
                    bad_accept_nod_ratio =  float(bad_accept_nod)/count_bad_nod
                    accept_nod_ratio = float(accept_nod)/(count_nod)
                    # if (bad_accept_nod_ratio > bad_accept_nod_ratio_max) \
#                         or (accept_nod_ratio < accept_nod_ratio_min):
#                         print 'nod BREAK'
#                         break
                    if (bad_accept_nod_ratio > bad_accept_nod_ratio_max):
                        #print 'too many accept'
                        break
                    if (accept_nod_ratio < accept_nod_ratio_min):
                        #print 'too many reject'
                        break
                    
                    if 0: #for debugging. 0 suppresses this for now.
                        print 'T: %.2e' % temperature, \
                            'accept nod ratio: %.2e ' %accept_nod_ratio, \
                            'bad accept nod ratio: %.2e' % bad_accept_nod_ratio, \
                            'energy: %.2e' % energy
         
        #print 'T: %.2e' % temperature, \
        #    'accept mod ratio: %.2e ' %accept_mod_ratio, \
        #    'bad accept mod ratio: %.2e' % bad_accept_mod_ratio, \
        #    'energy: %.2e' %energy, 'best: %.2e' %energy_best
        print 'T: %.2e' % temperature, \
            'energy: %.2e' %energy, 'best: %.2e' %energy_best
        temperature *= temp_scaling

    if extra_info:
        extra_dict = dict(energy = energy_array, temp = temp_array)
        return graph_partition, extra_dict
    else:
        return graph_partition

