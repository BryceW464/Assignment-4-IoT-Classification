#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (port) samples.
    Xd_ts : numpy array
           Array containing testing (port) samples.
    Xc_tr : numpy array
           Array containing training (port) samples.
    Xc_ts : numpy array
           Array containing testing (port) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Perform stage 1 of the classification procedure:
        train a random forest classifier using the NB prediction probabilities

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    pred : numpy array
           Final predictions on testing dataset.
    """
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1, oob_score=True)
    model.fit(X_tr, Y_tr)

    score = model.score(X_ts, Y_ts)
    print("RF accuracy = {}".format(score))

    pred = model.predict(X_ts)

    ################
    #2.1 (Decision Tree)

    max_depth = 0
    min_node = 0
    

    #To predict, navigate the tree using your test sample until you reach a leaf node.
    #The predicted class is the mode of the sample labels in that node.

    return pred

def decisionTree():
    #split dataset until each split reaches either:
        #max depth
        #number of samples in the group to split is less than min_node
        #optimal split results in a group with no samples (then use the parent node)
        #optimal split has a worse gini impurity than the parent node (use the parent node)
    pass


def child_node_gini_impurity(group_labels):
    #Pseudocode to find impurity of child nodes:
        #childScore = 1 - (number of each class present)^2
        #Example: Group 1 has 10 samples, of which 3 classes are present
        #5 samples in class 5, 3 samples in class 12, and 2 samples in class 29.
        # childScore = 1 - (0.5^2 + 0.3^2 + 0.2^2) = 0.62

    classDict = {}
    for label in group_labels:
        if label in classDict.keys():
            classDict[label] = classDict[label] + 1
        else:
            classDict[label] = 1

    baseImpurity = 1
    totalSampleCount = sum(classDict.values())
    for classCount in classDict.values():
        baseImpurity -= (classCount / totalSampleCount) ** 2

    return baseImpurity

def split_gini_impurity(group1Labels, group2Labels):
        #Pseudocode to find impurity of a split:
        #group1Count = number of samples in group 1
        #group2Count = number of samples in group 2
        #totalCount = total number of samples across both groups
        #G1 = impurity score of the left child node
        #G2 = impurity score of the right child node
        #splitScore = n1/n (G1) + n2/2 (G2)

    group1Count = len(group1Labels)
    group2Count = len(group2Labels)
    totalCount = group1Count + group2Count

    g1 = child_node_gini_impurity(group1Labels)
    g2 = child_node_gini_impurity(group2Labels)

    score = (group1Count / totalCount) * g1 + (group2Count / totalCount) * g2

    return score

def findSplitLocation(X_tr, X_ts, Y_tr, Y_ts):
    #For each feature, make a list of all values that make an appearance and order them.
    #Take the averages between each value and store them in a list for possible splits.
    #split the group data on each possible split into two groups.
    #Call split_gini_impurity given the two groups and get the gini impurity
    #the split with lowest gini impurity is chosen

    #    Parameters
    #----------
    #X_tr : numpy array
    #       Array containing training samples.
    #Y_tr : numpy array
    #       Array containing training labels.
    #X_ts : numpy array
    #       Array containing testing samples.
    #Y_ts : numpy array
    #       Array containing testing labels

    #for each feature, need all possible values across all samples. Those values need to get sorted.
    #then average between each point is thrown into possible splits and tested.

    allValues = set()
    possibleSplits = {}
    featureBestScores = {}
    featureBestSplits = {}

    #Looping through 12 times (there are 12 features in each sample)
    for i in range(12):

        #Looping through every sample in the training samples to get their values for the feature
        for sample in X_tr:
            allValues.add(sample[i])

        #Sorting allValues and getting possible split points by taking the average of every 2 adjacent points
        allValues = sorted(allValues)
        for y in range(len(allValues)-1):
            possibleSplits[(allValues[y] + allValues[y+1]) / 2] = 0

        #Creating the left and right child nodes. 
        #Then going through the training samples and adding each sample to either the left or right node depending if the feature value is < our split
        leftGroup = []
        rightGroup = []
        for split in possibleSplits:
            for x in range(len(X_tr)):
                if sample[i] < split:
                    leftGroup.append(Y_tr[x])
                else:
                    rightGroup.append(Y_tr[x])
            
            #the Gini impurity score is calculated for the split
            splitScore = split_gini_impurity(leftGroup, rightGroup)

            #that score is added to the dictionary for possibleSplits
            possibleSplits[split] = splitScore

            #Variables are cleared for next loop
            leftGroup.clear()
            rightGroup.clear()

        #Sort the possible splits : impurity scores dictionary to return a list with the lowest score at the front
        possibleSplits = sorted(possibleSplits)

        #Pulling the best score and split values and throwing them in the dictionary
        bestSplit = possibleSplits[0][0]
        bestScore = possibleSplits[0][1]
        featureBestScores[i] = bestScore
        featureBestSplits[i] = bestSplit

        #variables are cleared for the next loop    
        allValues.clear()
        possibleSplits.clear()

    #Sorting the best scores and returning the best feature, split, and impurity score
    featureBestScores = sorted(featureBestScores)
    bestFeature = featureBestScores[0][0]
    bestSplit = featureBestSplits[bestFeature]
    bestScore = featureBestScores[0][1]

    return bestFeature, bestSplit, bestScore


def main(args):
    """
    Perform main logic of program
    """
    # load dataset
    print("Loading dataset ... ")
    X, X_p, X_d, X_c, Y = load_data(args.root)

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # perform final classification
    print("Performing Stage 1 classification ... ")
    pred = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # print classification report
    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)
