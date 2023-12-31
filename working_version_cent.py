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
from algos import ConvexHull, minDpair, Weightedmidpt, Poincare_quantize, global_grouping, Mobius_add, Mobius_mul, Exp_map, Log_map, poincare_dist, point_on_geodesic,set_seeds, global_grouping_multi_labels_comm_spectral
from partition import part_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# import torch.optim as optim

plt.style.use('seaborn')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
device = 'cuda:1'


# def multi_hsvm_loss(w, X, y, C, loss_func):
#     c = X.size(0)
#     n = X.size(1)
#     d = X.size(2)
#     z = torch.zeros((n, c)).to(device)
#     for c_idx in range(c):
#         z[:, c_idx] = X[c_idx].mv(w[:, c_idx])
#     return C * loss_func(z, y) + w.pow(2).sum() / 2


# def hsvm_eval(w, X, y):
#     '''
#     input:
#         w: (d,c)
#         X: (c,n,d)
#         y: (n,), NOT one-hot
#     return:
#         acc: scalar
#     '''
#     c = X.size(0)
#     n = X.size(1)
#     pred_vals = torch.zeros((n, c)).to(device)
#     for c_idx in range(c):
#         pred_vals[:, c_idx] = X[c_idx].mv(w[:, c_idx])
#     pred = pred_vals.max(1)[1]
#     return pred.eq(y).float().mean()


# def tangent_hsvm_torch(X_train, train_labels, X_test, test_labels, C, p_arr, num_steps, curvature_const, opt_choice='LBFGS', lr=0.5, tol=1e-32, wd=0):
#     start = time.time()
#     n_classes = train_labels.max() + 1
#     n_train_samples = X_train.shape[0]
#     n_test_samples = X_test.shape[0]
#     X_train_log_map = []
#     X_test_log_map = []
    
#     # prepare the training data
#     for class_label in range(n_classes):
#         p = p_arr[class_label]
#         # train set
#         log_map_class = np.zeros_like(X_train, dtype=float)    
#         for i in range(n_train_samples):
#             log_map_class[i] = Log_map(X_train[i], p, curvature_const)
#         X_train_log_map.append(torch.from_numpy(log_map_class).unsqueeze(0))
        
#         # test set
#         log_map_class = np.zeros_like(X_test, dtype=float)
#         for i in range(n_test_samples):
#             log_map_class[i] = Log_map(X_test[i], p, curvature_const)
#         X_test_log_map.append(torch.from_numpy(log_map_class).unsqueeze(0))
    
#     X_train_log_map = torch.cat(X_train_log_map).float().to(device)
#     X_test_log_map = torch.cat(X_test_log_map).float().to(device)
#     y_train = torch.from_numpy(train_labels).to(device)
#     y_test = torch.from_numpy(test_labels).to(device)
    
#     loss_func = torch.nn.MultiMarginLoss().to(device)
    
#     # zero initialization
#     w = torch.autograd.Variable(torch.zeros(X_train.shape[1], n_classes).float().to(device), requires_grad=True)

#     def closure():
#         return multi_hsvm_loss(w, X_train_log_map, y_train, C, loss_func)
       
#     if opt_choice == 'LBFGS':
#         optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
#     elif opt_choice == 'Adam':
#         optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
#     else:
#         raise("Error: Not supported optimizer.")
    
#     best_loss = None
#     best_w = None
    
#     for _ in tqdm(range(num_steps)):
#         optimizer.zero_grad()
#         loss = multi_hsvm_loss(w, X_train_log_map, y_train, C, loss_func)
#         if best_loss is None or best_loss > loss:
#             best_loss = loss.clone().detach()
#             best_w = w.clone().detach()
#         loss.backward()
        
#         if opt_choice == 'LBFGS':
#             optimizer.step(closure)
#         elif opt_choice == 'Adam':
#             optimizer.step()
#         else:
#             raise("Error: Not supported optimizer.")
    
#     return hsvm_eval(best_w, X_test_log_map, y_test)


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


# def esvm_torch(X_train, train_labels, X_test, test_labels, C, p_arr, num_steps, opt_choice='LBFGS', lr=0.5, tol=1e-32, wd=0):
#     start = time.time()
#     n_classes = train_labels.max() + 1
#     n_train_samples = X_train.shape[0]
#     n_test_samples = X_test.shape[0]
    
#     X_train = torch.from_numpy(np.concatenate([X_train, np.ones((n_train_samples, 1))], axis=1)).float().to(device)  # add the freedom for bias
#     X_test = torch.from_numpy(np.concatenate([X_test, np.ones((n_test_samples, 1))], axis=1)).float().to(device)
#     y_train = torch.from_numpy(train_labels).to(device)
#     y_test = torch.from_numpy(test_labels).to(device)
    
#     loss_func = torch.nn.MultiMarginLoss().to(device)
    
#     # zero initialization
#     w = torch.autograd.Variable(torch.zeros(X_train.size(1), n_classes).float().to(device), requires_grad=True)

#     def closure():
#         return multi_esvm_loss(w, X_train, y_train, C, loss_func)
       
#     if opt_choice == 'LBFGS':
#         optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
#     elif opt_choice == 'Adam':
#         optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
#     else:
#         raise("Error: Not supported optimizer.")
    
#     best_loss = None
#     best_w = None
    
#     for _ in tqdm(range(num_steps)):
#         optimizer.zero_grad()
#         loss = multi_esvm_loss(w, X_train, y_train, C, loss_func)
#         if best_loss is None or best_loss > loss:
#             best_loss = loss.clone().detach()
#             best_w = w.clone().detach()
#         loss.backward()
        
#         if opt_choice == 'LBFGS':
#             optimizer.step(closure)
#         elif opt_choice == 'Adam':
#             optimizer.step()
#         else:
#             raise("Error: Not supported optimizer.")
    
#     return esvm_eval(best_w, X_test, y_test)


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
    parser.add_argument("--trails", type=int, default=5, help="Number of independent trials for each setting")
    parser.add_argument("--eps", type=float, default=1e-1, help="Quantization stepsize")
    parser.add_argument("--part_mode", type=str, default='iid', help="Partition fashion")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument('--save_fig', type=bool, default=False, help="Whether to save reference point figures")
    parser.add_argument('--save_path', type=str, default="neurips_results_final", help="Where to save results")
    parser.add_argument('--zero_intra_weights', type=bool, default=True, help="Whether to reduce the cross-convex hulls edges across local hulls from same client to small value")
    parser.add_argument('--multiclass', action='store_true', default=False, help="Multiclass")
    parser.add_argument('--CP', action='store_true', default=False, help="CP flag")
    parser.add_argument('--CE', action='store_true', default=False, help="CE flag")
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
    assert args.multiclass and  args.CE and args.CP
    R=np.linalg.norm(data_x, axis=1).max()
    if args.multiclass:  # multiclass
        # need to make data_y a 0-indexed array
        _, data_y = np.unique(data_y, return_inverse=True)
        
        # one row each for CE and CP
        acc = np.zeros((2,4,args.trails), dtype=float)
        
        for i in tqdm(range(args.trails)):
            # partition into training and testing sets
            data = {}
            data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(data_x, 
                                                                                                data_y, 
                                                                                                test_size=args.test_ratio, 
                                                                                                random_state=args.seed)

            i_type = 0
            for i_ch in [0,1]:
                for i_quant in [0,1]:
                    
                    data_x_tr = copy.deepcopy(data['x_train'])
                    data_y_tr = copy.deepcopy(data['y_train'])
                    data_x_label_wise={}
                    for i_label in range(len(interested_pairs)):
                        data_x_label_wise[i_label] = data_x_tr[data_y_tr==i_label,:]
                        
                        if i_ch==1:
                            data_temp=ConvexHull(data_x_label_wise[i_label].T, curvature_const=curve)
                            data_x_label_wise[i_label]=data_temp.T
                        if i_quant==1:
                            data_temp=Poincare_quantize(X=data_x_label_wise[i_label].T,curvature_const=curve,R=R,ep=args.eps)
                            data_x_label_wise[i_label]=data_temp.T
                            if i_ch==1:
                                # replicating how a client would do quantized convex hull
                                data_temp=ConvexHull(data_x_label_wise[i_label].T, curvature_const=curve)
                                data_x_label_wise[i_label]=data_temp.T

                    # combine data for the labels
                    data_x_final=np.vstack([data_x_label_wise[i_label] for i_label in range(len(interested_pairs))])
                    data_y_final=np.hstack([[i_label]*data_x_label_wise[i_label].shape[0] for i_label in range(len(interested_pairs))])
                    assert data_x_final.shape[0] == data_y_final.size  # data_x is of size n*2


                    if args.CP:
                        acc[0, i_type, i], _, _ = tangent_hsvm(data_x_final, data_y_final.astype(int), 
                                                                        data['x_test'], data['y_test'].astype(int), 
                                                                        C=0.1, multiclass=True, curvature_const=curve)

                    if args.CE:
                        # euclidean baseline
                        acc[1, i_type, i], _ = euclidean_svm(data_x_final, data_y_final.astype(int), 
                                                                data['x_test'], data['y_test'].astype(int), 
                                                                C=0.1, multiclass=True)
                    i_type += 1

            

    res_name = f'cent_acc_{args.dataset_name}_{args.num_clients}_{args.switch_ratio}_{args.ref_k}_{args.eps}_{args.part_mode}_{interested_pairs}'
    if args.multiclass:
        res_name += 'multi'
    else:
        res_name += 'binary'
    np.savez(f'{args.save_path}/{res_name}.npz', acc=acc)

    # print('centralized tangent hsvm:', np.mean(acc[0], axis=0), np.std(acc[0], axis=0))
    # print('centralized euclidean svm:', acc[1])
    # print('fl tangent hsvm:', acc[2])
    # print('fl euclidean svm:', acc[3])
    
    print("CP results")
    print('No CH-No Quant| No CH-Quant | CH-No Quant | CH-Quant')
    print('mean:')
    print(np.mean(acc[0], axis=1).T)
    
    print("CE results")
    print('No CH-No Quant| No CH-Quant | CH-No Quant | CH-Quant')
    print('mean:')
    print(np.mean(acc[1], axis=1).T)
