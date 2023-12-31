#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, confusion_matrix, f1_score
from numpy.linalg import norm
from sklearn.svm import LinearSVC
from GrahamScan import GrahamScan
from platt import *
from hsvm import *
from collections import Counter
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import random
import copy
import networkx as nx
from tabulate import tabulate
import ipdb
from bisect import bisect_left, insort_left
from itertools import combinations
from sklearn.cluster import SpectralClustering

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive']

def Parrallel_transport(v,p):
    """
    Parrallel transport of v in T_0 to T_p.
    """
    return (1-np.linalg.norm(p)**2)*v

def Poincare_Uniform_Data(N,d,gamma,R = 0.9, p = None, w = None, sample_type=None):
    """
    Generate points uniformly (in Euclidean sense) on the Poincare ball.
    N: Number of points.
    d: Dimension.
    gamma: Margin.
    R: Upper bound of the radius of points.
    p: Reference point. If not given, we randomly generate if within the ball of radius R.
    w: Normal vector of ground truth linear classifier. If not given, we generate it uniformly at random on the ball of radius R.
    sample_type: whether to sample in Eculidean space "E" or in hyperbolic space "H"
    """
    assert sample_type is not None, "the sampling type has to be specified"
    assert d==2, "other d not configured"
    if sample_type=="E":
        # Points within uniform ball
        X = np.random.randn(d,N) # values randomly drawn from standard normal
        r = np.random.rand(N)**(1/d) # values drawn randomly from uniform and then raised to 1/d 
        c = r/np.sqrt(np.sum(X**2,0))
        X = X*c*R
    elif sample_type=="H":
        R_H=np.log((1+R)/(1-R))
        r_H=2*np.arcsinh(np.sqrt(np.random.rand(N))*np.sinh(R_H/2))
        r_new=(np.exp(r_H)-1)/(np.exp(r_H)+1)
        ang=2*np.pi*np.random.rand(N)
        x1=np.multiply(r_new,np.cos(ang))
        x2=np.multiply(r_new,np.sin(ang))
        X=np.zeros((d,N))
        X[0,:]=x1
        X[1,:]=x2        
    else:
        assert False, "not configured"
    # Construct true classifier
    if p is None:
        p = np.random.randn(d)
        p = p/np.linalg.norm(p)*np.random.rand(1)*R
    
    if w is None:
        w = np.random.randn(d)
    
    # the normal vector needs to be moved from tangent space at origin to p in a parallel way
    w = Parrallel_transport(w,p)
    w = w/np.linalg.norm(w)
    
    # Define true labels
    y = np.zeros(N)
    for n in range(N):
        if np.dot(w,Log_map(X[:,n],p))> 0:
            y[n] = 1
            
        else:
            y[n] = -1
    
    # Remove points which invalidates margin assumption
    lmda_p = 2/(1-np.linalg.norm(p)**2)
    for n in range(N):
        xn = X[:,n]
        vn = Log_map(xn,p)
        # the following condition is just re-written form of eq (14) in KAIS,
        # and the fact that inner product and label have same sign due to way of data creation
        # need to modify if being applied for real dataset
        if np.dot(w,vn)*y[n] < np.sinh(gamma)*np.linalg.norm(vn)*(1-np.tanh(lmda_p*np.linalg.norm(vn)/2)**2)/np.tanh(lmda_p*np.linalg.norm(vn)/2)/2:
            y[n] = 0
            X[:,n] = 0
            
    idx = np.argwhere(np.all(X[..., :] == 0, axis=0))
    X = np.delete(X, idx, axis=1)
    y = np.delete(y, idx)
    N = len(y)
    
    return N,X,y,w,p


def uniform_data_gen(N,d,gamma,R,a_r,counts,seed_v,cluster_thres=5000,sample_type="E"):
    r"""
    function to generate the synthetic datasets
    N: total points to sample
    d: 2
    gamma: margin for removing the points
    R: maximum bound for the points in th Poincare disc
    a_r: reference point for data generation to be sampled as a_R*R/5
    counts: number of datasets to generate with this configuration
    cluster_thres: minimum size of any label
    seed: random seed
    sample_type: "E" for uniform in Euclidean space, "H" for uniform in hyperbolic space
    """
    assert d==2, "not configured for higher dimensions"
    set_seeds(seed_v)

    for i in range(counts):

        Flag = False
        while not Flag:
            p_r = a_r/5*R
            p_theta = 2*np.pi*np.random.random((1,))[0]
            p = np.array([p_r*np.cos(p_theta),p_r*np.sin(p_theta)])
            # Generating points uniformly on Poincare ball.
            # Can set the reference point p that the ground truth classifier pass through.
            _,X,y,w,p = Poincare_Uniform_Data(N,d,gamma,R = R, p = p, sample_type=sample_type)
            X=X.T
            # This ensures that resulting label is not too unbalanced (5% at least). Can be modified or removed if desired.
            if (len(y[y==1])>cluster_thres) and (len(y[y==-1])>cluster_thres):
                Flag = True
        print(f"i: {i}, X_shape: {X.shape}, X_pos: {X[y==1,:].shape}, X_neg: {X[y==-1,:].shape}")
        np.savez(f'embedding/uniform_{sample_type}_{R}_{N}_{a_r}_{gamma}_{i}.npz', data=X, label=y)


def quantize_eu(X,R,ep):

    r"""
    given a set of points in dimension*points format, the goal is to find the quantized points, using grid quantization
    """
    # output: quantized points in Euclidean space
    assert ep<R, "unexpected quantizaton resolution"
    assert X.shape[0]==2, "code not suitable for other dimensions"
    x_coord=X[0,:]
    y_coord=X[1,:]
    N_R=np.ceil(R/ep)

    # quantize x-coordinate
    n_x=np.floor((N_R/R)*x_coord)+0.5
    x_coord_new=(R/N_R)*n_x

    # quantize y-coordinate
    n_y=np.floor((N_R/R)*y_coord)+0.5
    y_coord_new=(R/N_R)*n_y

    Xnew=np.zeros_like(X)
    Xnew[0,:]=x_coord_new
    Xnew[1,:]=y_coord_new
    # print("N_t: {0}, R_H: {1}, N_R_H: {2}".format(N_t,R_H,N_R_H))
    return Xnew



def Poincare_quantize(X,curvature_const,R=0.99,ep=0.0001,tol=0.0001):

    r"""
    given a set of points in dimension*points format, the goal is to find the quantized points, using approach 2 
    described in our paper
    """
    # done: check whether the angle is actually between 0 and 2 pi
    # done: check that arctan2 is being used correctly, as first argument is y coordinate
    # required inputs: featureset X, max_Poincare_radius R, hyperbolic distance bound ep, curvature_const
    # output: quantized points in the Poincare disk
    assert curvature_const>0, "Code applibale only to -curvature_const hyperbolic space"
    R+=tol
    s_cc=1/np.sqrt(curvature_const)
    R_H=s_cc*np.log((s_cc+R)/(s_cc-R))
    x1=X[1,:]
    x2=X[0,:]
    ang=np.arctan2(x1, x2)
    ang[ang<0]=2*np.pi+ang[ang<0]
    # print(ang, ang/(2*np.pi)*360)
    r=np.sqrt(np.power(x1,2)+np.power(x2,2))
    r_H=s_cc*np.log((s_cc+r)/(s_cc-r))
    
    # quantize radius
    N_R_H=np.ceil(2*R_H/ep)
    n_r=np.floor((N_R_H/R_H)*r_H)+0.5
    r_H_new=(R_H/N_R_H)*n_r
    r_new=s_cc*(np.exp(r_H_new/s_cc)-1)/(np.exp(r_H_new/s_cc)+1)

    # quantize angle
    Circum=2*np.pi*s_cc*np.sinh(R_H/s_cc) 
    N_t=np.ceil(2*Circum/ep)
    n_t=np.floor((N_t/(2*np.pi))*ang)+0.5
    ang_new=(2*np.pi/N_t)*n_t 

    x1new=np.multiply(r_new,np.sin(ang_new))
    x2new=np.multiply(r_new,np.cos(ang_new))

    Xnew=np.zeros_like(X)
    Xnew[1,:]=x1new
    Xnew[0,:]=x2new
    # print("N_t: {0}, R_H: {1}, N_R_H: {2}".format(N_t,R_H,N_R_H))
    return Xnew

def global_grouping_multi_labels_comm_spectral(XClients,yClients,curvature_const,exact_labels,eps,R,prs=1,zero_intra_weights=True):
    r"""
    ***Module to be used only when there are more than 2 global labels***
    Pipeline for Convex Hulls Computation, Quantization of Convex Hulls, Graph based Grouping of Convex Hulls
    XClients: Dictionary containing the features for each client, key is client_ID starting from 0
    yClients: Dictionary containing the labels for each client
    curvature_const: Poincare disc has a curvature of -curvature_const
    exact_labels: numpy array containing the unique global labels, contains int values
    eps: Poincare Quantization parameter
    prs: pseudo-random seed
    zero_intra_weights: flag variable if True will make the cross edges in any client much smaller

    """
    set_seeds(prs)     
    # Number of clients
    L=len(XClients) 
    # Obtain the labels
    # exact_labels=(np.unique(yClients[0])).astype(int)
    total_labels=len(exact_labels)
    assert total_labels>2, "Some error in client's labels"
    # obtain the convex hulls
    convexHullsClients={}
    for iClient in range(L):
        # key=(client_ID,label)
        for iLabel in range(total_labels):
            convexHullsClients[(iClient,exact_labels[iLabel])]=ConvexHull(XClients[iClient][:,yClients[iClient]==exact_labels[iLabel]],curvature_const)
    # # quantize and communicate the convex hulls
    for iClient in range(L):
        for iLabel in range(total_labels):
            # # quantize the data points in convex hull
            convexHullsClients[(iClient,exact_labels[iLabel])]=Poincare_quantize(X=convexHullsClients[(iClient,exact_labels[iLabel])],curvature_const=curvature_const,R=R,ep=eps)
            # # remove repeated entries, as for any bin, we send just one boundary point
            # convexHullsClients[(iClient,exact_labels[iLabel])]=(np.vstack(list({tuple(row) for row in convexHullsClients[(iClient,exact_labels[iLabel])].T}))).T
            # # need to re-compute convex hull for illustration as some points might be moved too much or removed completely
            # note that in actual implementation, this step is not needed
            convexHullsClients[(iClient,exact_labels[iLabel])]=ConvexHull(convexHullsClients[(iClient,exact_labels[iLabel])],curvature_const)
    # # Proposed graph based global aggregation of convex hulls
    # get all hulls in one dictionary
    X_Hulls_all={}
    X_Hulls_all_grnd_trth_lbls={}
    total_hulls=int(L*total_labels)
    i_hull=0
    for iClient in range(L):
        for iLabel in range(total_labels):
            X_Hulls_all[i_hull]=convexHullsClients[(iClient,exact_labels[iLabel])]
            X_Hulls_all_grnd_trth_lbls[i_hull]=exact_labels[iLabel]
            i_hull+=1
    assert len(X_Hulls_all)==total_hulls, "some logical issue"

    # compute pairwise distances between each pair of convex hulls
    distances={}
    min_distance=np.inf
    second_min_distance=np.inf
    for i_hull_1 in range(total_hulls-1):
        for i_hull_2 in range(i_hull_1+1,total_hulls):
            distances[(i_hull_1,i_hull_2)]=convex_hulls_distance(X_Hulls_all[i_hull_1],X_Hulls_all[i_hull_2],curvature_const=curvature_const)
            if distances[(i_hull_1,i_hull_2)]<min_distance:
                min_distance=distances[(i_hull_1,i_hull_2)]
            elif distances[(i_hull_1,i_hull_2)]>min_distance and distances[(i_hull_1,i_hull_2)]<second_min_distance:
                second_min_distance=distances[(i_hull_1,i_hull_2)]


    # print("distances")
    # print(distances)
    # create edges with weight as 1/distance 
    G = nx.Graph()
    min_weight=np.inf
    max_weight=-1
    for i_hull_1 in range(total_hulls-1):
        for i_hull_2 in range(i_hull_1+1,total_hulls):
            if distances[(i_hull_1,i_hull_2)]==0:
                weight=10*1/(second_min_distance)
            else:
                weight=1/distances[(i_hull_1,i_hull_2)]
            G.add_edge(i_hull_1, i_hull_2, weight=weight)
            if weight<min_weight:
                min_weight=weight
            if weight>max_weight:
                max_weight=weight
    # print("min_weight: {0}, max_weight: {1}".format(min_weight,max_weight))
    # if wanted, can also make sure the weight of edges between pairwise convex hulls from same client is 1/100 of minimum edge weight,
    # further forcing the hulls from same client to be in separate groups in partition
    if zero_intra_weights:
        for iClient in range(L):
            # extract the corresponding indices for convex hulls in X_Hulls_all
            client_hulls_indices=[int(i_hull) for i_hull in range(iClient*total_labels,(iClient+1)*total_labels)]
            # then, enumerate all the pairwise combinations from the list 
            client_all_pairs=list(combinations(client_hulls_indices, 2))
            for i_pair in client_all_pairs:
                G[i_pair[0]][i_pair[1]]['weight']=0.01*min_weight
    # run the partitioning algorithm
    nodelist=list(range(len(G.nodes()))) 
    adj_mat = nx.to_numpy_matrix(G,nodelist=nodelist)
    # print(G.nodes())
    # print(adj_mat)
    sc = SpectralClustering(total_labels, affinity='precomputed', n_init=100, assign_labels="discretize")
    sc.fit(np.asarray(adj_mat))
    partition_assignments=sc.labels_
    subsetIDs=np.unique(partition_assignments)
    # print("\ntotal communities: {0}".format(len(subsetIDs)))
    try:
        assert len(subsetIDs)==total_labels, "Data not suitable for partitioning"
    except:
        print("\ntotal communities: {0}".format(len(subsetIDs)))
        assert False
    # for each set in the partition, compute the corresponding majority ground truth label as key and the 
    # aggregated convex hulls as value 
    convexHullsGlobal={}
    convexHullsLabelingGlobal={}

    for i_partition in range(total_labels):
        temp_hulls_indxs=list(np.where(partition_assignments==i_partition)[0])
        # print("i_partition: {0}, temp_hulls_indxs: ".format(i_partition),temp_hulls_indxs)
        # for i_hull in temp_hulls_indxs:
        #     print(i_hull)
        temp_hulls_labels=(np.array([X_Hulls_all_grnd_trth_lbls[i_hull] for i_hull in temp_hulls_indxs])).astype(int)
        temp_values, temp_counts = np.unique(temp_hulls_labels, return_counts=True)
        temp_global_label=temp_values[temp_counts == temp_counts.max()][0]
        # print("temp_global_label: ",temp_global_label)
        
        if temp_global_label in convexHullsGlobal:
            convexHullsLabelingGlobal[temp_global_label]=np.hstack((convexHullsLabelingGlobal[temp_global_label],np.array(temp_hulls_indxs).astype(int)))
            for i_hull in temp_hulls_indxs:
                convexHullsGlobal[temp_global_label]=np.hstack((convexHullsGlobal[temp_global_label],X_Hulls_all[i_hull]))            
        else:
            convexHullsLabelingGlobal[temp_global_label]=np.array(temp_hulls_indxs).astype(int)
            convexHullsGlobal[temp_global_label]=[]
            for i_hull in temp_hulls_indxs:
                convexHullsGlobal[temp_global_label].append(X_Hulls_all[i_hull])
            convexHullsGlobal[temp_global_label]=np.hstack(convexHullsGlobal[temp_global_label])
    for i_global_label in convexHullsGlobal:
        # remove repeated entries
        convexHullsGlobal[i_global_label]=(np.vstack(list({tuple(row) for row in convexHullsGlobal[i_global_label].T}))).T
    return X_Hulls_all,convexHullsLabelingGlobal,convexHullsGlobal

def global_grouping(XClients, yClients, curvature_const, eps=0.001, R=0.9991, prs=1, ref_k=3):
    r"""
    Pipeline for Convex Hulls Computation, Quantization of Convex Hulls, Graph based Grouping of Convex Hulls
    XClients: Dictionary containing the features for each client, key is client_ID starting from 0
    yClients: Dictionary containing the labels for each client
    eps: Poincare Quantization parameter
    R: Maximum length of embedding in the Poincare disc, default value of 0.9991 is for Fashion-MNIST 
    prs: pseudo-random seed
    """
    set_seeds(prs)
    
    # Number of clients
    L=len(XClients)
    # Obtain the labels
    exact_labels=(np.unique(yClients[0])).astype(int)
    assert len(exact_labels)==2, "Some error in client's labels"
    # obtain the convex hulls
    convexHullsClients={}
    for iClient in range(L):
    # key=(client_ID,label)
        
        # debug
        # np.save('tmp_data.npy', XClients[iClient][:,yClients[iClient]==exact_labels[0]])
        
        convexHullsClients[(iClient,exact_labels[0])]=ConvexHull(XClients[iClient][:,yClients[iClient]==exact_labels[0]],curvature_const)
        convexHullsClients[(iClient,exact_labels[1])]=ConvexHull(XClients[iClient][:,yClients[iClient]==exact_labels[1]],curvature_const)
    # # quantize and communicate the convex hulls
    for iClient in range(L):
        # # quantize the data points in convex hull
        convexHullsClients[(iClient,exact_labels[0])]=Poincare_quantize(X=convexHullsClients[(iClient,exact_labels[0])],curvature_const=curvature_const,R=R,ep=eps)
        # # remove repeated entries, as for any bin, we send just one boundary point
        # convexHullsClients[(iClient,exact_labels[0])]=(np.vstack(list({tuple(row) for row in convexHullsClients[(iClient,exact_labels[0])].T}))).T
        # # need to re-compute convex hull for illustration as some points might be moved too much or removed completely
        # note that in actual implementation, this step is not needed
        convexHullsClients[(iClient,exact_labels[0])]=ConvexHull(convexHullsClients[(iClient,exact_labels[0])],curvature_const)
        
        convexHullsClients[(iClient,exact_labels[1])]=Poincare_quantize(X=convexHullsClients[(iClient,exact_labels[1])],curvature_const=curvature_const,R=R,ep=eps)
        # convexHullsClients[(iClient,exact_labels[1])]=(np.vstack(list({tuple(row) for row in convexHullsClients[(iClient,exact_labels[1])].T}))).T
        convexHullsClients[(iClient,exact_labels[1])]=ConvexHull(convexHullsClients[(iClient,exact_labels[1])],curvature_const)

    # # Proposed graph based global aggregation of convex hulls
    # get all hulls in one dictionary
    X_Hulls_all={}
    total_hulls=int(L*2)
    i_hull=0
    for iClient in range(L):
        X_Hulls_all[i_hull]=convexHullsClients[(iClient,exact_labels[0])]
        X_Hulls_all[i_hull+1]=convexHullsClients[(iClient,exact_labels[1])]
        i_hull+=2
    assert len(X_Hulls_all)==total_hulls, "some logical issue"

    # compute pairwise distances between each pair of convex hulls
    distances={}
    min_distance=np.inf
    second_min_distance=np.inf
    for i_hull_1 in range(total_hulls-1):
        for i_hull_2 in range(i_hull_1+1,total_hulls):
            distances[(i_hull_1,i_hull_2)]=convex_hulls_distance(X_Hulls_all[i_hull_1],X_Hulls_all[i_hull_2],curvature_const=curvature_const)
            if distances[(i_hull_1,i_hull_2)]<min_distance:
                min_distance=distances[(i_hull_1,i_hull_2)]
            elif distances[(i_hull_1,i_hull_2)]>min_distance and distances[(i_hull_1,i_hull_2)]<second_min_distance:
                second_min_distance=distances[(i_hull_1,i_hull_2)]
    


    # print("min_distance: {0}, second_min_distance: {1}".format(min_distance,second_min_distance))
    # create edges with weight as 1/distance 
    G = nx.Graph()
    min_weight=np.inf
    max_weight=-1
    for i_hull_1 in range(total_hulls-1):
        for i_hull_2 in range(i_hull_1+1,total_hulls):
            if distances[(i_hull_1,i_hull_2)]==0:
                weight=10*1/(second_min_distance)
            else:
                weight=1/distances[(i_hull_1,i_hull_2)]
            G.add_edge(i_hull_1, i_hull_2, weight=weight)
            if weight<min_weight:
                min_weight=weight
            if weight>max_weight:
                max_weight=weight
    # print("min_weight: {0}, max_weight: {1}".format(min_weight,max_weight))
    # if wanted, can also make sure the weight of edges between pairwise convex hulls from same client is 1/100 of minimum edge weight
    zero_weights=False
    if zero_weights:
        for iClient in range(L):
            G[int(iClient*2)][int(iClient*2+1)]['weight']=0.01*min_weight
    # run the partitioning algorithm
    max_iter=10
    # prs=1
    partition=nx.algorithms.community.kernighan_lin_bisection(G=G,max_iter=max_iter,seed=prs)
    convexHullsGlobal={}
    # need to assign global labels in line with test dataset
    # assigning global label as a majority of the ground truth local labels
    # by construction, even indexed hulls in X_Hulls_all have exact_labels[0] 
    list_global_label_0_hulls=list(partition[0])
    list_global_label_1_hulls=list(partition[1])
    only_odd=[num for num in list_global_label_0_hulls if num % 2 == 1]
    odd_count=len(only_odd)
    only_even=[num for num in list_global_label_0_hulls if num % 2 == 0]
    even_count=len(only_even)
    if even_count>=odd_count:
        global_label_0=exact_labels[0]
        global_label_1=exact_labels[1]
    else:
        global_label_0=exact_labels[1]
        global_label_1=exact_labels[0]
    # obtain the features
    convexHullsGlobal[global_label_0]=[]
    convexHullsGlobal[global_label_1]=[]
    for i_len in range(len(partition[0])):
        convexHullsGlobal[global_label_0].append(X_Hulls_all[list_global_label_0_hulls[i_len]])
    for i_len in range(len(partition[1])):
        convexHullsGlobal[global_label_1].append(X_Hulls_all[list_global_label_1_hulls[i_len]])
    
    # as we are doing SVM directly at the server side, let's keep all data in the global convex hulls
    # that belong to any of the associated local convex hulls 
    convexHullsGlobal[global_label_0]=np.hstack(convexHullsGlobal[global_label_0])
    convexHullsGlobal[global_label_0]=(np.vstack(list({tuple(row) for row in convexHullsGlobal[global_label_0].T}))).T
    convexHullsGlobal[global_label_1]=np.hstack(convexHullsGlobal[global_label_1])
    convexHullsGlobal[global_label_1]=(np.vstack(list({tuple(row) for row in convexHullsGlobal[global_label_1].T}))).T
    # print("shape 0: ",convexHullsGlobal[global_label_0].shape)
    # print("shape 1: ",convexHullsGlobal[global_label_1].shape)
    # print("wait")
    # time.sleep(3)
    CH0=ConvexHull(convexHullsGlobal[global_label_0],curvature_const)
    CH1=ConvexHull(convexHullsGlobal[global_label_1],curvature_const)
    min_dist_pair=minDpair(CH0, CH1, k=ref_k,curvature_const=curvature_const)
    # pref = Weightedmidpt(min_dist_pair[:,0],min_dist_pair[:,1],0.5)

    return convexHullsClients, convexHullsGlobal, min_dist_pair, partition


def ccw(p0, p1, p2, curvature_const):
    """
    Outer product in hyperbolic Graham scan.
    """
    v01 = Log_map(p1, p0, curvature_const) / norm(Log_map(p1, p0, curvature_const))
    v12 = Log_map(p2, p0, curvature_const) / norm(Log_map(p2, p0, curvature_const))
    return v01[0] * v12[1] - v01[1] * v12[0]


def ConvexHull(X, curvature_const):
    """
    Finding convexhull in Poincare disk using hyperbolic version of Graham scan.
    """
#     input: X is a d*n matrix
#     Assume d = 2 so far
    
    # Ensure no duplicate
    X = (np.vstack(list({tuple(row) for row in X.T}))).T
    if X.shape[1]<=3:
        return X
    
#     Step1: Find the point furthest from origin
    R_list = np.linalg.norm(X,axis=0)
#     Check if multiple maximum, pick arbitrary.
    idx = np.argwhere(R_list==np.amax(R_list))
    if len(idx)>1:
        idx=idx[0]
    idx = np.squeeze(idx)
    pstart = X[:,idx]
    p0 = pstart
    origin = np.zeros((2,))
#     Sort points by inner angle with p0
    logX = np.zeros(np.shape(X))
    Iplist = np.zeros(np.shape(X)[1])
    logX[:,idx] = np.zeros((2,))
    logX_norm = np.zeros(np.shape(X)[1])
    normal_vec = -Log_map(origin, p0, curvature_const) / np.linalg.norm(Log_map(origin, p0, curvature_const))
    tangent_vec = np.array([-normal_vec[1],normal_vec[0]])
    for n in range(np.shape(X)[1]):
        if n == idx:
            continue
        logX[:,n] = Log_map(X[:,n], p0, curvature_const)
        logX_norm[n] = np.linalg.norm(logX[:,n])
        Iplist[n] = np.dot(logX[:,n]/np.linalg.norm(logX[:,n]),tangent_vec)
          
#     Make sure p0 is sorted as the last point
    Iplist[idx] = -np.inf
    logX_norm[idx] = 0
#     Sort Iplist
    Ipidx = np.flip(np.argsort(Iplist))
    
    first_ptidx = 0
    Points = X[:,Ipidx]
    
    d = np.shape(X)[0]
    N = np.shape(X)[1]
    Stack = np.zeros((d,N+1))
    Stack[:,0] = pstart
    end_idx = 0
    for point in Points.T:
        while (end_idx>0) and (ccw(Stack[:,end_idx-1],Stack[:,end_idx],point,curvature_const)<0):
            end_idx -= 1
            
        end_idx += 1
        Stack[:,end_idx] = point
        
    return Stack[:,:(end_idx+1)]


def set_seeds(seed=1):
    np.random.seed(seed)
    random.seed(seed)

    
def convex_hulls_distance(cxHull_1, cxHull_2, curvature_const):
    total_points_hull_1 = cxHull_1.shape[1]
    total_points_hull_2 = cxHull_2.shape[1]
    distance = 0
    for i_1 in range(total_points_hull_1):
        for i_2 in range(total_points_hull_2):
            distance += poincare_dist(x=cxHull_1[:, i_1], y=cxHull_2[:, i_2], curvature_const=curvature_const)
    distance = distance / (total_points_hull_1 * total_points_hull_2)
    return distance

def convex_hulls_distance_eu(cxHull_1, cxHull_2):
    total_points_hull_1 = cxHull_1.shape[1]
    total_points_hull_2 = cxHull_2.shape[1]
    distance = 0
    for i_1 in range(total_points_hull_1):
        for i_2 in range(total_points_hull_2):
            distance += dist_eu(x=cxHull_1[:, i_1], y=cxHull_2[:, i_2])
    distance = distance / (total_points_hull_1 * total_points_hull_2)
    return distance

def Mobius_add(x, y, curvature_const):
    DeNom = 1 + 2 * curvature_const * np.dot(x, y) + curvature_const**2 * norm(x)**2 * norm(y)**2
    Nom = (1 + 2 * curvature_const * np.dot(x, y) + curvature_const * norm(y)**2) * x + (1 - curvature_const * norm(x)**2) * y
    return Nom / DeNom


def Mobius_mul(r, x, curvature_const):
    return np.tanh(r * np.arctanh(np.sqrt(curvature_const) * norm(x))) * x / norm(x) / np.sqrt(curvature_const)


def Exp_map(v, p, curvature_const):
    lbda = 2 / (1 - curvature_const * norm(p)**2)
    temp = np.tanh(np.sqrt(curvature_const) * lbda * norm(v) / 2) * v / np.sqrt(curvature_const) / norm(v)
    return Mobius_add(p, temp, curvature_const)


def Log_map(x, p, curvature_const):
    lbda = 2 / (1 - curvature_const * norm(p)**2)
    temp = Mobius_add(-p, x, curvature_const)
    if norm(temp) == 0:
        return temp
    else:
        return 2 / np.sqrt(curvature_const) / lbda * np.arctanh(np.sqrt(curvature_const) * norm(temp)) * temp / norm(temp)


def poincare_dist(x, y, curvature_const):
    return 2 / np.sqrt(curvature_const) * np.arctanh(np.sqrt(curvature_const) * norm(Mobius_add(-x, y, curvature_const)))
    # return np.arccosh(1 + 2*(norm(x-y)**2)/(1-norm(x)**2)/(1-norm(y)**2))

def dist_eu(x, y):
    return math.sqrt((x[1]-y[1])**2 + (x[0]-y[0])**2)

def point_on_geodesic(x, y, t, curvature_const):
    return Exp_map(t * Log_map(y, x, curvature_const), x, curvature_const)


def Weightedmidpt(C1, C2, t, curvature_const):
    """
    Compute the weighted midpoint from C1 to C2 in Poincare ball. t is the time where t=0 we get C1 and t=1 we get C2.
    """
    if np.allclose(C1, C2):
        return C1
    else:
        return point_on_geodesic(C1, C2, t, curvature_const)


def minDpair(CH1, CH2, k, curvature_const):
    """
    Finding the minimum k distance pairs for convex hull CH1 and CH2 in Poincare disk.
    """
    N1 = np.shape(CH1)[1]
    N2 = np.shape(CH2)[1]
    cur_minD = []
    for n1 in range(N1):
        for n2 in range(N2):
            dist = poincare_dist(CH1[:, n1], CH2[:, n2], curvature_const)
            insort_left(cur_minD, (dist, n1, n2))
            if len(cur_minD) > k:
                cur_minD.pop()
                assert len(cur_minD) == k
    output = np.zeros((k, 2, 2))
    for idx in range(k):
        output[idx, :, 0] = CH1[:, cur_minD[idx][1]]
        output[idx, :, 1] = CH2[:, cur_minD[idx][2]]
    return output

def plotgeodesic(p0,curvature_const,p1=None,v=None,option='segment',plotColor='k',linestyle='solid',linewidth=2):

    assert option in ['segment','p2p_line','pv_line']
    if option is 'segment':
    #     default use 100 point
        t = np.linspace(0,1,100)
        output = np.zeros((2,100))
        for i in range(100):
            output[:,i] = Weightedmidpt(p0,p1,t[i],curvature_const=curvature_const)
    #     ipdb.set_trace()
        plt.plot(output[0,:],output[1,:],c=plotColor,linestyle=linestyle,linewidth=linewidth)
        return None
    else:
#         Assume that 2 end points need to pass the circle R = 0.99
        R = 0.99
        if option is 'p2p_line':
            v = Mobius_add(-p0,p1,curvature_const=curvature_const)
            v = v/np.linalg.norm(v)
        
        v = np.array(v)
        t = np.linspace(0,1,100)
        Line = np.zeros((2,100))
        for n in range(100):
            Line[:,n] = Exp_map(v*t[n],p0,curvature_const=curvature_const)
        Line[:,0] = p0
        AdLine = np.zeros((2,100))
        count = 1.0
        while np.linalg.norm(Line[:,-1])<R:
            for n in range(100):
                AdLine[:,n] = Exp_map(v*(t[n]+count),p0,curvature_const=curvature_const)
            Line = np.append(Line,AdLine,axis= 1)
            count += 1.
        plt.plot(Line[0,:],Line[1,:],c=plotColor,linestyle=linestyle,linewidth=linewidth)
        
        Line = np.zeros((2,100))
        for n in range(100):
            Line[:,n] = Exp_map(-v*t[n],p0,curvature_const=curvature_const)
        Line[:,0] = p0
        count = 1.0
        while np.linalg.norm(Line[:,-1])<R:
            for n in range(100):
                AdLine[:,n] = Exp_map(-v*(t[n]+count),p0,curvature_const=curvature_const)
            Line = np.append(Line,AdLine,axis= 1)
            count += 1.
            
        plt.plot(Line[0,:],Line[1,:],c=plotColor,linestyle=linestyle,linewidth=linewidth)
        return None
        
