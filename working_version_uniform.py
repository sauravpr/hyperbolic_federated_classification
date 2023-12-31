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
from algos import ConvexHull, minDpair, Weightedmidpt, global_grouping
from partition import part_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
plt.style.use('seaborn')

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']


def Mobius_add(x, y, c=1.):
    DeNom = 1 + 2*c*np.dot(x, y) + (c**2)*(norm(x)**2)*(norm(y)**2)
    Nom = (1 + 2*c*np.dot(x, y) + c*(norm(y)**2))*x + (1 - c*(norm(x)**2))*y
    return Nom/DeNom


def Exp_map(v, p, c=1.):
    lbda = 2/(1-c*np.dot(p,p))
    temp = np.tanh(np.sqrt(c)*lbda*np.sqrt(np.dot(v, v))/2)*v/np.sqrt(c)/np.sqrt(np.dot(v, v))
    return Mobius_add(p, temp,c)


def Log_map(x, p, c=1.):
    lbda = 2/(1-c*np.dot(p,p))
    temp = Mobius_add(-p, x, c)
    if norm(temp) == 0:
        return temp
    else:
        return 2/np.sqrt(c)/lbda * np.arctanh(np.sqrt(c)*norm(temp)) * temp / norm(temp)


def poincare_dist(x, y):
    return np.arccosh(1 + 2*(norm(x-y)**2)/(1-norm(x)**2)/(1-norm(y)**2))


def point_on_geodesic(x, y, t):
    return Exp_map(t*Log_map(y, x), x)


def zero_based_labels(y):
    labels = list(np.unique(y))
    new_y = [labels.index(y_val) for y_val in y]
    return np.array(new_y)


def plot_geodesic_old(p, v):
    # p is the reference point and v is the speed vector perpendicular to w
    max_R = 0.999
    t = np.linspace(0, 3, 500)
    geo_line = np.zeros((500, 2))
    count = 0
    for i in range(500):
        if t[i] == 0:
            tmp = p
        else:
            tmp = Exp_map(v*t[i], p)
        if norm(tmp) > max_R:
            break
        geo_line[count, :] = tmp
        count += 1
    return geo_line[0: count, :]


def plot_geodesic_new(p0, v, ax, c):
    R = 0.999
    t = np.linspace(0, 1, 100)
    # pos
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(v * t[n], p0)
    Line[:, 0] = p0
    AdLine = np.zeros((2, 100))
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)
    # neg
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(-v * t[n], p0)
    Line[:, 0] = p0
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(-v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)


def Eval(X, y, y_pred, p=None, w1=None, xi=None, C=None):
    if p is not None:
        d = np.shape(X)[0]
        N = np.shape(X)[1]
        lmda_p = 2 / (1 - np.linalg.norm(p) ** 2)
        Z = np.zeros((d, N))
        I = np.identity(d)
        for n in range(N):
            vn = Log_map(X[:, n], p)
            etan = 2 * np.tanh(lmda_p * np.linalg.norm(vn) / 2) / np.linalg.norm(vn) / (
                        1 - np.tanh(lmda_p * np.linalg.norm(vn) / 2) ** 2)
            Z[:, n] = etan * vn

        y_hat1 = np.zeros(N)
        y_hat2 = np.zeros(N)
        decision_val = np.zeros(N)
        for n in range(N):
            if (w1 is not None):
                y_hat1[n] = np.sign(np.dot(w1, Log_map(X[:, n], p)))
                if y_hat1[n] != y[n]:
                    print(n, y_hat1[n], y[n])
                decision_val[n] = np.dot(w1, Log_map(X[:, n], p))

        if (w1 is not None):
            return np.sum(y * y_hat1 > 0) / N * 100, decision_val

def tangent_hsvm_synthetic(X_train, train_labels, X_test, test_labels, C, p=None, p_arr=None, multiclass=False, curvature_const=1.0,max_iter=1000):
    # the labels need to be 0-based indexed
    assert multiclass is False, "module only suitable for binary classification"
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
            pos_coords = []
            neg_coords = []
            binarized_labels = []
            for i in range(n_train_samples):
                if train_labels[i] == class_label:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(1)
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(-1)
            if len(pos_coords) <= 1:
                # skip very small classes
                continue
            binarized_labels = np.array(binarized_labels)
            # # convex hull of positive cluster
            # pos_hull = GrahamScan(pos_coords)
            # neg_hull = GrahamScan(neg_coords)
            # print('Convex hull generated!')
            # # get the reference point p by finding the min dis pair
            # p = np.zeros(2)
            # min_dis = float('inf')
            # break_flag = False
            # for i in range(pos_hull.shape[0]):
            #     for j in range(neg_hull.shape[0]):
            #         if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
            #             min_dis = poincare_dist(pos_hull[i], neg_hull[j])
            #             p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
            #             if min_dis < 1e-2:
            #                 break_flag = True
            #                 break
            #         print('\r{}/{}, {}/{}, {}'.format(i, pos_hull.shape[0], j, neg_hull.shape[0], min_dis), end='')
            #     if break_flag:
            #         break
            # print('\nreference point p found!')
            p = p_arr[class_label]
            # map training data using log map
            X_train_log_map = np.zeros_like(X_train, dtype=float)
            for i in range(n_train_samples):
                X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
            # print('log transformation done!')
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=False,max_iter=max_iter)
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
    else:
        # if there if only two classes, no need for Platt probability
        # if p is given, use the given p, else first estimate p
        if p is None:
            pos_coords = []
            neg_coords = []
            for i in range(n_train_samples):
                if train_labels[i] == 1:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
            # convex hull of positive cluster
            pos_hull = GrahamScan(pos_coords)
            neg_hull = GrahamScan(neg_coords)
            # get the reference point p by finding the min dis pair
            p = np.zeros(2)
            min_dis = float('inf')
            for i in range(pos_hull.shape[0]):
                for j in range(neg_hull.shape[0]):
                    if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
                        min_dis = poincare_dist(pos_hull[i], neg_hull[j])
                        p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
        # we have p now
        X_train_log_map = np.zeros_like(X_train, dtype=float)
        for i in range(n_train_samples):
            X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=False,max_iter=max_iter)
        linear_svm.fit(X_train_log_map, train_labels)
        X_test_log_map = np.zeros_like(X_test, dtype=float)
        for i in range(n_test_samples):
            X_test_log_map[i] = Log_map(X_test[i], p, curvature_const)
        y_pred = linear_svm.predict(X_test_log_map)
        y_pred_train = linear_svm.predict(X_train_log_map)
    return accuracy_score(train_labels, y_pred_train)*100, accuracy_score(test_labels, y_pred)*100, time.time() - start, linear_svm


# def cho_hsvm(X_train, train_labels, X_test, test_labels, C, multiclass=False, max_epoches=30):
#     # fit multiclass hsvm and get prediction accuracy
#     start = time.time()
#     n_train_samples = X_train.shape[0]
#     hsvm_clf = LinearHSVM(early_stopping=2, C=C, num_epochs=max_epoches, lr=0.001, verbose=True,
#                           multiclass=multiclass, batch_size=int(n_train_samples/50))
#     hsvm_clf.fit(poincare_pts_to_hyperboloid(X_train, eps=1e-6, metric='minkowski'), train_labels)
#     y_pred = hsvm_clf.predict(poincare_pts_to_hyperboloid(X_test, eps=1e-6, metric='minkowski'))
#     return accuracy_score(test_labels, y_pred)*100, time.time() - start


def euclidean_svm_synthetic(X_train, train_labels, X_test, test_labels, C, multiclass, max_iter=1000):
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
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=True,max_iter=max_iter)
            linear_svm.fit(X_train, binarized_labels)
            # print('binary SVM done!')
            w = linear_svm.coef_[0]
            bi = linear_svm.intercept_[0]
            decision_vals = np.array([np.dot(w, x)+bi for x in X_train])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # print('Platt probability computed!')
            # map testing data using log map
            for i in range(n_test_samples):
                test_decision_val = np.dot(w, X_test[i])+bi
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)
            # print('Probability for each test samples computed!')
        y_pred = np.argmax(test_probability, axis=1)
    else:
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=True,max_iter=max_iter)
        linear_svm.fit(X_train, train_labels)
        y_pred = linear_svm.predict(X_test)
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments on real data")
    parser.add_argument("--dataset_name", type=str, default='uc_stromal', help="Which dataset to test")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients in FL setting")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--switch_ratio", type=float, default=0.0, help="The ratio of clients with switched labels")
    parser.add_argument("--ref_k", type=int, default=5, help="Number of candidates for reference points")
    parser.add_argument("--C", type=int, default=20000, help="Larger value means more penalty for errors")
    parser.add_argument("--max_iter_SVM", type=int, default=200000, help="As name suggests")
    parser.add_argument("--a_r", type=int, default=None, help="fraction representation for hyperplane distance from origin")
    parser.add_argument("--R", type=float, default=None, help="Poincare Euclidean bound for data points")
    parser.add_argument("--eps", type=float, default=1e-1, help="Quantization stepsize")
    parser.add_argument("--gamma", type=float, default=None, help="margin the dataset")
    parser.add_argument("--part_mode", type=str, default='iid', help="Partition fashion")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--trails", type=int, default=25, help="How many trails to run")
    parser.add_argument('--save_fig', type=bool, default=False, help="Whether to save reference point figures")
    parser.add_argument('--save_path', type=str, default="neurips_results_final", help="Where to save results")
    args = parser.parse_args()
    print(args)
    start_time = time.time()
    assert args.a_r is not None
    assert args.gamma is not None
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    acc = np.zeros((4, args.trails), dtype=float)
    time_used = np.zeros((4, args.trails), dtype=float)

    for i in tqdm(range(args.trails)):
        # load data 
        # print(f"i: {i}")
        N=int(args.a_r*20000)
        data_path = f'embedding/{args.dataset_name}_{args.R}_{N}_{args.a_r}_{args.gamma}_{i}.npz'
        data_npz = np.load(data_path)
        data_x, data_y = data_npz['data'], data_npz['label']
        assert len(np.unique(data_y))==2, "not configured for multiple labels"
        assert data_x.shape[0] == data_y.size  # data_x is of size n*2

        data_class_pos = copy.deepcopy(data_x[data_y==1, :])
        data_class_neg = copy.deepcopy(data_x[data_y==-1, :])
        data_pairwise = np.concatenate([data_class_pos, data_class_neg], axis=0)
        label_pairwise = np.array([0]*data_class_pos.shape[0] + [1]*data_class_neg.shape[0], dtype=int)

        # partition into training and testing sets
        data = {}
        data['x_train'], data['x_test'], data['y_train'], data['y_test'] = train_test_split(data_pairwise, 
                                                                                            label_pairwise, 
                                                                                            test_size=args.test_ratio, 
                                                                                            random_state=i*(args.seed+1))

        ### first do centralized setting

        # compute the reference point first
        CH_class1 = ConvexHull(data['x_train'][data['y_train'] == 0, :].T)
        CH_class2 = ConvexHull(data['x_train'][data['y_train'] == 1, :].T)
        # Find k min dist pairs on these two convex hull
        # now MDP are of size (k, 2, 2)
        MDP = minDpair(CH_class1, CH_class2, k=args.ref_k)
        best_train_acc, best_p = 0, None
        for p_idx in range(args.ref_k):
            # choose p as their mid point
            cur_p = Weightedmidpt(MDP[p_idx, :, 0], MDP[p_idx, :, 1], 0.5)
            # run tangent_hsvm with multiclass=False
            cur_train_acc, cur_test_acc, cur_time, _ = tangent_hsvm_synthetic(data['x_train'], data['y_train'].astype(int), 
                                                                    data['x_test'], data['y_test'].astype(int), 
                                                                    C=args.C, 
                                                                    p=cur_p, 
                                                                    multiclass=False,
                                                                    max_iter=args.max_iter_SVM)
            if cur_train_acc > best_train_acc:
                best_train_acc = cur_train_acc
                acc[0, i] = cur_test_acc
                time_used[0, i] = cur_time
                best_p = cur_p
        assert best_p is not None

        # # save the figure
        # if args.save_fig:
        #     fig_path = f'{args.save_path}/{args.dataset_name}_{interested_pairs[j]}_{interested_pairs[k]}_centralized.pdf'
        #     plot_binary_with_p(data_class1, data_class2, best_p, fig_path, label1=type_list[interested_pairs[j]], label2=type_list[interested_pairs[k]])

        # euclidean baseline
        acc[1, i], time_used[1, i] = euclidean_svm_synthetic(data['x_train'], 
                                                                data['y_train'].astype(int), 
                                                                data['x_test'],
                                                                data['y_test'].astype(int), 
                                                                C=args.C,
                                                                max_iter=args.max_iter_SVM, 
                                                                multiclass=False)

        ### then do FL setting

        # partition the training data to clients
        XClients, yClients, R = part_data(data['x_train'], data['y_train'], 
                                            num_clients=args.num_clients,
                                            switch_ratio=args.switch_ratio,
                                            part_mode=args.part_mode, seed=i*(args.seed+1))

        # perform global grouping of convex hulls
        convexHullsClients, _, MDP_global, partition = global_grouping(XClients, yClients, eps=args.eps, R=R, prs=i*(args.seed+1), ref_k=args.ref_k)
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

        # try different global reference point candidates
        best_train_acc, best_p_global = 0, None
        for p_idx in range(args.ref_k):
            cur_p = Weightedmidpt(MDP_global[p_idx, :, 0], MDP_global[p_idx, :, 1], 0.5)
            # run tangent_hsvm with multiclass=False
            cur_train_acc, cur_test_acc, cur_time, _ = tangent_hsvm_synthetic(data_agg, label_agg, 
                                                                    data['x_test'], data['y_test'].astype(int),
                                                                    C=args.C, 
                                                                    p=cur_p,
                                                                    max_iter=args.max_iter_SVM,
                                                                    multiclass=False)
            if cur_train_acc > best_train_acc:
                best_train_acc = cur_train_acc
                acc[2, i] = cur_test_acc
                time_used[2, i] = cur_time
                best_p_global = cur_p
        assert best_p_global is not None

        # # save the figure
        # if args.save_fig:
        #     fig_path = f'{args.save_path}/{args.dataset_name}_{interested_pairs[j]}_{interested_pairs[k]}_fl.pdf'
        #     if even_count >= len(partition[0]) / 2:
        #         plot_binary_with_p(data_class1_agg, data_class2_agg, best_p_global, fig_path, label1=type_list[interested_pairs[j]], label2=type_list[interested_pairs[k]])
        #     else:
        #         plot_binary_with_p(data_class2_agg, data_class1_agg, best_p_global, fig_path, label1=type_list[interested_pairs[j]], label2=type_list[interested_pairs[k]])

        # fl euclidean svm baseline
        acc[3, i], time_used[3, i] = euclidean_svm_synthetic(data_agg, 
                                                            label_agg, 
                                                            data['x_test'],
                                                            data['y_test'].astype(int), 
                                                            C=args.C,
                                                            max_iter=args.max_iter_SVM, 
                                                            multiclass=False)

    np.savez(f'{args.save_path}/acc_{args.dataset_name}_{args.R}_{N}_{args.a_r}_{args.gamma}_{args.num_clients}_{args.switch_ratio}_{args.ref_k}_{args.eps}_{args.part_mode}.npz', acc=acc, time_used=time_used)

    print('centralized tangent hsvm:', acc[0])
    print('centralized euclidean svm:', acc[1])
    print('fl tangent hsvm:', acc[2])
    print('fl euclidean svm:', acc[3])
    print('Time used:', time.time() - start_time)