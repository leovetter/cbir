def query():

    W = pickle.load(open('output/pascal/8-cluster/oasis_weights.npy', 'rb'))

    features_names_and_labels = pickle.load(open('../algos/bag/output/pascal/8-cluster/train_features.npy', 'rb'))
    train_features = []
    for triplet in features_names_and_labels:
        train_features.append(triplet['features'])
    train_features = np.array(features)

    precomp = np.dot(W, train_features.T)

    val_features = 

query()
