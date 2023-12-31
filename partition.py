# In this script we always assume that there are only two classes for label switching purposes

import numpy as np


def part_data(data, label, num_clients, switch_ratio=0.1, part_mode='iid', num_classes=None, seed=None):
    '''
        Partition data points to each client either in an iid / non-iid fashion.
    '''
    np.random.seed(seed=seed)
    permute_order = np.random.permutation(label.size)
    data = data[permute_order, :]
    label = label[permute_order]

    if not num_classes:
        num_classes = np.unique(label).size
    assert data.shape[0] >= num_classes * num_clients, "Number of data points is smaller than number of clients"

    client_data = {}
    for j in range(num_clients):
        client_data[j] = {'data': [], 'label': []}

    for i in range(num_classes):
        active = np.where(label == i)[0]
        count_i = active.size

        if part_mode == 'iid':
            count_each_client = int(np.ceil(count_i / num_clients))
            for j in range(num_clients):
                client_data[j]['data'].append(data[active[j * count_each_client: min((j + 1) * count_each_client, count_i)], :])
                client_data[j]['label'].extend([i] * client_data[j]['data'][-1].shape[0])
        else:
            # np.random.seed(seed=seed)
            while True:
                count_partition = np.random.permutation(list(range(1, count_i)))[:num_clients+1]
                count_partition.sort()
                count_partition[0] = 0
                count_partition[-1] = count_i

                # we need to make sure each cluster on each client has at least 4 points
                diff = np.diff(count_partition)
                if diff.min() >= 4:
                    break
            
            for j in range(num_clients):
                client_data[j]['data'].append(data[active[count_partition[j]: count_partition[j+1]], :])
                client_data[j]['label'].extend([i] * (count_partition[j+1] - count_partition[j]))
    
    XClients = []
    yClients = []
    switch_count = int(num_clients * switch_ratio)
    for j in range(num_clients):
        client_data[j]['data'] = np.concatenate(client_data[j]['data'], axis=0)
        client_data[j]['label'] = np.array(client_data[j]['label'], dtype=int)
        # first switch_count clients get labels inverted
        if j < switch_count:
            client_data[j]['label'] = 1 - client_data[j]['label']
        
        assert client_data[j]['data'].shape[0] == client_data[j]['label'].size
        # shuffle the client data
        np.random.seed(seed=seed)
        permute_order = np.random.permutation(client_data[j]['label'].size)
        client_data[j]['data'] = client_data[j]['data'][permute_order, :]
        client_data[j]['label'] = client_data[j]['label'][permute_order]

        XClients.append(client_data[j]['data'].T)
        yClients.append(client_data[j]['label'])
    
    return XClients, yClients, np.linalg.norm(data, axis=1).max()
