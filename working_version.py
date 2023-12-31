import os
import math
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVC
from GrahamScan import GrahamScan
import matplotlib.pyplot as plt
from platt import *
import time
from hsvm import *
import argparse
from algos import ConvexHull, minDpair, Weightedmidpt, global_grouping, Mobius_add, Mobius_mul, Exp_map, Log_map, poincare_dist, point_on_geodesic,set_seeds, global_grouping_multi_labels_comm_spectral
from partition import part_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# import torch.optim as optim

plt.style.use('seaborn')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
device = 'cuda:1'


def multi_esvm_loss(w, X, y, C, loss_func):
    z = X.mm(w).to(device)
    return C * loss_func(z, y) + w.pow(2).sum() / 2


def esvm_eval(w, X, y):
    '''
    input:
        w: (d,c)
        X: (c,n,d)
        y: (n,), NOT one-hot
    return:
        acc: scalar
    '''
    pred_vals = X.mm(w)
    pred = pred_vals.max(1)[1]
    return pred.eq(y).float().mean()


def tangent_hsvm(X_train, train_labels, X_test, test_labels, C, curvature_const, p=None, p_arr=None, multiclass=False):
    # the labels need to be 0-based indexed
    start = time.time()
    n_classes = train_labels.max() + 1
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    if multiclass:
        store_p = False
        if p_arr is None:
            p_arr = np.zeros((n_classes, X_train.shape[1]))
            store_p = True
        # there is more than 2 classes, using ovr strategy
        # find optimal p for each ovr classifier
        test_probability = np.zeros((n_test_samples, n_classes), dtype=float)
        for class_label in range(n_classes):
            if store_p:
                pos_coords = []
                neg_coords = []
                binarized_labels = []
                for i in range(n_train_samples):
                    if train_labels[i] == class_label:
                        pos_coords.append([X_train[i][0], X_train[i][1]])
                        binarized_labels.append(1)
                    else:
                        neg_coords.append([X_train[i][0], X_train[i][1]])
                        binarized_labels.append(-1)
                if len(pos_coords) <= 1:
                    # skip very small classes
                    continue
                pos_coords = np.array(pos_coords)
                neg_coords = np.array(neg_coords)
                binarized_labels = np.array(binarized_labels)

                CH_class1 = ConvexHull(pos_coords.T, curvature_const=curvature_const)
                CH_class2 = ConvexHull(neg_coords.T, curvature_const=curvature_const)
                # Find k min dist pairs on these two convex hull
                # now MDP are of size (k, 2, 2)
                MDP = minDpair(CH_class1, CH_class2, k=args.ref_k, curvature_const=curvature_const)
                
                best_train_acc, best_p = 0, None
                for p_idx in range(args.ref_k):
                    # choose p as their mid point
                    cur_p = Weightedmidpt(MDP[p_idx, :, 0], MDP[p_idx, :, 1], 0.5, curvature_const)
                    # run tangent_hsvm with multiclass=False
                    cur_train_acc, _, _ = tangent_hsvm(X_train, binarized_labels, 
                                                       X_train, binarized_labels, 
                                                       C=C, p=cur_p, multiclass=False, curvature_const=curvature_const)
                    if cur_train_acc > best_train_acc:
                        best_train_acc = cur_train_acc
                        best_p = cur_p
                
                assert best_p is not None
                p_arr[class_label] = best_p.copy()
                p = best_p.copy()
            else:
                p = p_arr[class_label]
            
            # map training data using log map
            X_train_log_map = np.zeros_like(X_train, dtype=float)
            for i in range(n_train_samples):
                X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
            # print('log transformation done!')
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, max_iter=10000, fit_intercept=False)
            linear_svm.fit(X_train_log_map, binarized_labels)
            # print('binary SVM done!')
            w = linear_svm.coef_[0]
            decision_vals = np.array([np.dot(w, x) for x in X_train_log_map])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # print('Platt probability computed!')
            # map testing data using log map
            for i in range(n_test_samples):
                x_test_log_map = Log_map(X_test[i], p, curvature_const)
                test_decision_val = np.dot(w, x_test_log_map)
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)
            # print('Probability for each test samples computed!')
        y_pred = np.argmax(test_probability, axis=1)
        # TODO: compute y_train for multiclass case
        
        return accuracy_score(test_labels, y_pred)*100, time.time() - start, p_arr
    else:
        # if there if only two classes, no need for Platt probability
        # if p is given, use the given p, else first estimate p
        assert p is not None  # p should be given outside
        # if p is None:
        #     pos_coords = []
        #     neg_coords = []
        #     for i in range(n_train_samples):
        #         if train_labels[i] == 1:
        #             pos_coords.append((X_train[i][0], X_train[i][1]))
        #         else:
        #             neg_coords.append((X_train[i][0], X_train[i][1]))
        #     # convex hull of positive cluster
        #     pos_hull = GrahamScan(pos_coords)
        #     neg_hull = GrahamScan(neg_coords)
        #     # get the reference point p by finding the min dis pair
        #     p = np.zeros(2)
        #     min_dis = float('inf')
        #     for i in range(pos_hull.shape[0]):
        #         for j in range(neg_hull.shape[0]):
        #             if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
        #                 min_dis = poincare_dist(pos_hull[i], neg_hull[j])
        #                 p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
        # # we have p now
        X_train_log_map = np.zeros_like(X_train, dtype=float)
        for i in range(n_train_samples):
            X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, max_iter=10000, fit_intercept=False)
        linear_svm.fit(X_train_log_map, train_labels)
        X_test_log_map = np.zeros_like(X_test, dtype=float)
        for i in range(n_test_samples):
            X_test_log_map[i] = Log_map(X_test[i], p, curvature_const)
        y_pred = linear_svm.predict(X_test_log_map)
        y_pred_train = linear_svm.predict(X_train_log_map)
        return accuracy_score(train_labels, y_pred_train)*100, accuracy_score(test_labels, y_pred)*100, time.time() - start


def cho_hsvm(X_train, train_labels, X_test, test_labels, C, multiclass=False, max_epoches=30):
    # fit multiclass hsvm and get prediction accuracy
    start = time.time()
    n_train_samples = X_train.shape[0]
    hsvm_clf = LinearHSVM(early_stopping=2, C=C, num_epochs=max_epoches, lr=0.001, verbose=True,
                          multiclass=multiclass, batch_size=int(n_train_samples/50))
    hsvm_clf.fit(poincare_pts_to_hyperboloid(X_train, eps=1e-6, metric='minkowski'), train_labels)
    y_pred = hsvm_clf.predict(poincare_pts_to_hyperboloid(X_test, eps=1e-6, metric='minkowski'))
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


def euclidean_svm(X_train, train_labels, X_test, test_labels, C, multiclass=False):
    # make sure the data is centered by the center of X_train
    # x_mean = np.mean(X_train, axis=0)
    # for i in range(X_train.shape[0]):
    #     X_train[i] -= x_mean
    # for i in range(X_test.shape[0]):
    #     X_test[i] -= x_mean
    
    start = time.time()
    n_classes = train_labels.max() + 1
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    if multiclass:
        # there is more than 2 classes, using ovr strategy
        # find optimal p for each ovr classifier
        test_probability = np.zeros((n_test_samples, n_classes), dtype=float)
        for class_label in range(n_classes):
            # print('Processing class:', class_label)
            binarized_labels = []
            for i in range(n_train_samples):
                if train_labels[i] == class_label:
                    binarized_labels.append(1)
                else:
                    binarized_labels.append(-1)
            binarized_labels = np.array(binarized_labels)
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, max_iter=10000, fit_intercept=True)
            linear_svm.fit(X_train, binarized_labels)
            # print('binary SVM done!')
            w = linear_svm.coef_[0]
            b = linear_svm.intercept_[0]
            decision_vals = np.array([np.dot(w, x) + b for x in X_train])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # print('Platt probability computed!')
            # map testing data using log map
            for i in range(n_test_samples):
                test_decision_val = np.dot(w, X_test[i]) + b
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)
            # print('Probability for each test samples computed!')
        y_pred = np.argmax(test_probability, axis=1)
    else:
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, max_iter=10000, fit_intercept=True)
        linear_svm.fit(X_train, train_labels)
        y_pred = linear_svm.predict(X_test)
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments on real data")
    parser.add_argument("--dataset_name", type=str, default='uc_stromal', help="Which dataset to test")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients in FL setting")
    parser.add_argument("--interested_pairs_comb", type=int, default=0, help="Which label combination to select")
    parser.add_argument("--test_ratio", type=float, default=0.4, help="Test ratio")
    parser.add_argument("--switch_ratio", type=float, default=0.0, help="The ratio of clients with switched labels")
    parser.add_argument("--ref_k", type=int, default=3, help="Number of candidates for reference points")
    parser.add_argument("--eps", type=float, default=1e-1, help="Quantization stepsize")
    parser.add_argument("--part_mode", type=str, default='iid', help="Partition fashion")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--trails", type=int, default=10, help="How many trails to run")
    parser.add_argument('--save_fig', type=bool, default=False, help="Whether to save reference point figures")
    parser.add_argument('--save_path', type=str, default="neurips_results_final", help="Where to save results")
    parser.add_argument('--zero_intra_weights', type=bool, default=True, help="Whether to reduce the cross-convex hulls edges across local hulls from same client to small value")
    parser.add_argument('--multiclass', action='store_true', default=False, help="Multiclass")
    parser.add_argument('--CP', action='store_true', default=False, help="CP flag")
    parser.add_argument('--CE', action='store_true', default=False, help="CE flag")
    parser.add_argument('--FLP', action='store_true', default=False, help="FLP flag")
    parser.add_argument('--FLE', action='store_true', default=False, help="FLE flag")
    args = parser.parse_args()
    print(args)
    set_seeds(args.seed)
    start_time = time.time()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # load data

    if args.dataset_name == 'olsson':
        data_path = 'embedding/olsson_poincare_embedding.npz'
        type_list = list(range(8))
        # interested_pairs = [0, 1, 2, 3, 4, 5]
        interested_pairs_list = [list(range(8))]
        interested_pairs = interested_pairs_list[args.interested_pairs_comb]
    
    print(f"interested_pairs: {interested_pairs}")

    # load the data
    data_npz = np.load(data_path)
    if args.dataset_name.startswith('olsson') and 'eu' not in args.dataset_name or args.dataset_name.startswith('fashion') or args.dataset_name.startswith('cifar'):
        data_x = np.concatenate([data_npz['x_train'], data_npz['x_test']], axis=0)
        data_y = np.concatenate([data_npz['y_train'], data_npz['y_test']])
        curve = 1.0
    else:
        data_x, data_y, curve = data_npz['data'], data_npz['label'], data_npz['curve']
    
    assert data_x.shape[0] == data_y.size  # data_x is of size n*2
    # extract data for interested pairs only
    active = []
    for i in range(data_y.size):
        if int(data_y[i]) in interested_pairs:
            active.append(i)
    data_x = data_x[active, :]
    data_y = data_y[active]
    
    if args.multiclass:  # multiclass
        # need to make data_y a 0-indexed array
        _, data_y = np.unique(data_y, return_inverse=True)
        
        acc = np.zeros((4, args.trails), dtype=float)
        time_used = np.zeros((4, args.trails), dtype=float)
        
        for i in tqdm(range(args.trails)):
            # partition into training and testing sets
            data = {}
            data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(data_x, 
                                                                                                data_y, 
                                                                                                test_size=args.test_ratio, 
                                                                                                random_state=args.seed)

            ### first do centralized setting

            if args.CP:
                acc[0, i], time_used[0, i], p_arr = tangent_hsvm(data['x_train'], data['y_train'].astype(int), 
                                                                 data['x_test'], data['y_test'].astype(int), 
                                                                 C=0.1, multiclass=True, curvature_const=curve)

            if args.CE:
                # euclidean baseline
                acc[1, i], time_used[1, i] = euclidean_svm(data['x_train'], data['y_train'].astype(int), 
                                                           data['x_test'], data['y_test'].astype(int), 
                                                           C=0.1, multiclass=True)
            
            ## TODO: FL part

            if args.FLP or args.FLE:
                label_list = list(np.unique(data['y_train']).astype(int))

                # partition the training data to clients
                XClients, yClients, R = part_data(data['x_train'], data['y_train'], 
                                                    num_clients=args.num_clients,
                                                    switch_ratio=args.switch_ratio,
                                                    part_mode=args.part_mode, seed=args.seed*i)

                # perform global grouping of convex hulls
                _,convexHullsLabelingGlobal,convexHullsGlobal=global_grouping_multi_labels_comm_spectral(XClients,yClients,
                                                                                                         curvature_const=curve,
                                                                                                         exact_labels=label_list,
                                                                                                         eps=args.eps,R=R,
                                                                                                         prs=i*(args.seed+1),
                                                                                                         zero_intra_weights=args.zero_intra_weights)
                print("convexHullsLabelingGlobal: ",convexHullsLabelingGlobal)
                # construct the dataset suitable for the hsvm
                # labels may not be ordered in the keys list
                total_labels=len(label_list)
                data_agg = [None]*total_labels
                label_agg = [None]*total_labels
                for i_label in list(convexHullsGlobal.keys()):
                    data_agg[i_label] = convexHullsGlobal[i_label].T
                    label_agg[i_label] = [i_label]*data_agg[i_label].shape[0]
                for i_label in label_list:
                    # adding dummy values for missing labels as a heuristic 
                    if data_agg[i_label] is None:
                        data_agg[i_label]=np.zeros((1,2))
                        label_agg[i_label]=[i_label]
                data_agg=np.vstack(data_agg)
                label_agg=np.hstack(label_agg)

            if args.FLP:
                acc[2, i], time_used[2, i], p_arr = tangent_hsvm(data_agg, label_agg, 
                                                                 data['x_test'], data['y_test'].astype(int), 
                                                                 C=0.1, multiclass=True, curvature_const=curve)

            if args.FLE:
                acc[3, i], time_used[3, i] = euclidean_svm(data_agg, label_agg, 
                                                           data['x_test'], data['y_test'].astype(int), 
                                                           C=0.1, multiclass=True)


    else:  # binary
        acc = np.zeros((4, args.trails, math.comb(len(interested_pairs), 2)), dtype=float)
        time_used = np.zeros((4, args.trails, math.comb(len(interested_pairs), 2)), dtype=float)


        for i in tqdm(range(args.trails)):
            comb_count = 0
            for j in range(len(interested_pairs)):
                for k in range(j+1, len(interested_pairs)):
                    # print(interested_pairs[j], interested_pairs[k])

                    active = np.where(data_y == interested_pairs[j])[0]
                    data_class1 = data_x[active, :]
                    active = np.where(data_y == interested_pairs[k])[0]
                    data_class2 = data_x[active, :]
                    data_pairwise = np.concatenate([data_class1, data_class2], axis=0)
                    label_pairwise = np.array([0]*data_class1.shape[0] + [1]*data_class2.shape[0], dtype=int)

                    # partition into training and testing sets
                    data = {}
                    data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(data_pairwise, 
                                                                                                        label_pairwise, 
                                                                                                        test_size=args.test_ratio, 
                                                                                                        random_state=args.seed)

                    ### first do centralized setting

                    if args.CP:
                        # compute the reference point first
                        CH_class1 = ConvexHull(data['x_train'][data['y_train'] == 0, :].T, curve)
                        CH_class2 = ConvexHull(data['x_train'][data['y_train'] == 1, :].T, curve)
                        # Find k min dist pairs on these two convex hull
                        # now MDP are of size (k, 2, 2)
                        MDP = minDpair(CH_class1, CH_class2, k=args.ref_k, curvature_const=curve)
                        best_train_acc, best_p = 0, None
                        for p_idx in range(args.ref_k):
                            # choose p as their mid point
                            cur_p = Weightedmidpt(MDP[p_idx, :, 0], MDP[p_idx, :, 1], 0.5, curve)
                            # run tangent_hsvm with multiclass=False
                            cur_train_acc, cur_test_acc, cur_time = tangent_hsvm(data['x_train'], data['y_train'].astype(int), 
                                                                                 data['x_test'], data['y_test'].astype(int), 
                                                                                 C=5, p=cur_p, multiclass=False, curvature_const=curve)
                            if cur_train_acc > best_train_acc:
                                best_train_acc = cur_train_acc
                                acc[0, i, comb_count] = cur_test_acc
                                time_used[0, i, comb_count] = cur_time
                                best_p = cur_p
                        assert best_p is not None


                    if args.CE:
                        # euclidean baseline
                        acc[1, i, comb_count], time_used[1, i, comb_count] = euclidean_svm(data['x_train'].copy(), 
                                                                                           data['y_train'].astype(int), 
                                                                                           data['x_test'].copy(),
                                                                                           data['y_test'].astype(int), 
                                                                                           C=5, multiclass=False)

                    ### then do FL setting
                    if args.FLP or args.FLE:
                        # partition the training data to clients
                        XClients, yClients, R = part_data(data['x_train'], data['y_train'], 
                                                          num_clients=args.num_clients,
                                                          switch_ratio=args.switch_ratio,
                                                          part_mode=args.part_mode, seed=args.seed*i)

                        # perform global grouping of convex hulls
                        convexHullsClients, _, MDP_global, partition = global_grouping(XClients, yClients, curvature_const=curve,
                                                                                       eps=args.eps, R=R, prs=args.seed, ref_k=args.ref_k)
                        # concatenate local CHs
                        data_class1_agg = []
                        even_count = 0
                        for node_idx in partition[0]:
                            data_class1_agg.append(convexHullsClients[node_idx // 2, node_idx % 2].T)
                            if node_idx % 2 == 0:
                                even_count += 1
                        data_class1_agg = np.concatenate(data_class1_agg, axis=0)

                        data_class2_agg = []
                        for node_idx in partition[1]:
                            data_class2_agg.append(convexHullsClients[node_idx // 2, node_idx % 2].T)
                        data_class2_agg = np.concatenate(data_class2_agg, axis=0)

                        if even_count >= len(partition[0]) / 2:
                            label_agg = [0] * data_class1_agg.shape[0] + [1] * data_class2_agg.shape[0]
                        else:
                            label_agg = [1] * data_class1_agg.shape[0] + [0] * data_class2_agg.shape[0]

                        data_agg = np.concatenate([data_class1_agg, data_class2_agg], axis=0)
                        label_agg = np.array(label_agg, dtype=int)

                    if args.FLP:
                        # try different global reference point candidates
                        best_train_acc, best_p_global = 0, None
                        for p_idx in range(args.ref_k):
                            cur_p = Weightedmidpt(MDP_global[p_idx, :, 0], MDP_global[p_idx, :, 1], 0.5, curve)
                            # run tangent_hsvm with multiclass=False
                            cur_train_acc, cur_test_acc, cur_time = tangent_hsvm(data_agg, label_agg, 
                                                                                 data['x_test'], data['y_test'].astype(int),
                                                                                 C=5, p=cur_p, multiclass=False, curvature_const=curve)
                            if cur_train_acc > best_train_acc:
                                best_train_acc = cur_train_acc
                                acc[2, i, comb_count] = cur_test_acc
                                time_used[2, i, comb_count] = cur_time
                                best_p_global = cur_p
                        assert best_p_global is not None

                    if args.FLE:
                        # fl euclidean svm baseline
                        acc[3, i, comb_count], time_used[3, i, comb_count] = euclidean_svm(data_agg.copy(), 
                                                                                           label_agg, 
                                                                                           data['x_test'].copy(),
                                                                                           data['y_test'].astype(int), 
                                                                                           C=5, multiclass=False)

                    # move to the next pairwise combination
                    comb_count += 1
    
    res_name = f'acc_{args.dataset_name}_{args.num_clients}_{args.switch_ratio}_{args.ref_k}_{args.eps}_{args.part_mode}_{interested_pairs}'
    if args.multiclass:
        res_name += 'multi'
    else:
        res_name += 'binary'
    np.savez(f'{args.save_path}/{res_name}.npz', acc=acc, time_used=time_used)
    
    print('CP | CE | FLP | FLE')
    print('mean:')
    print(np.mean(acc, axis=1).T)
    
    print('std:')
    print(np.std(acc, axis=1).T)
    
    print('Time used:', time.time() - start_time)