def informationGain(featurelabels):
    total = featurelabels.shape[0]
    ones = featurelabels[featurelabels[featurelabels.columns[0]] == 1].shape[0]
    zeros = featurelabels[featurelabels[featurelabels.columns[0]] == 0].shape[0]
    parentEntropy = entropyCalculator(featurelabels[['Class']])
    entropyChildWithOne = entropyCalculator(featurelabels[featurelabels[featurelabels.columns[0]] == 1][['Class']])
    entropyChildWithZero = entropyCalculator(featurelabels[featurelabels[featurelabels.columns[0]] == 0][['Class']])
#     print ("left entropy : " + str(entropyChildWithZero))
#     print ("right entropy : " + str(entropyChildWithOne))
    infoGain = parentEntropy - (ones/total)*entropyChildWithOne - (zeros/total)*entropyChildWithZero
    return infoGain