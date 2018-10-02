import operator
from math import log


#Assumption 1: Last column of dataSet contains the class label to which thta entry belongs.
#Assumption 2: All enteries have same number of columns.


def calcShannonEnt(dataSet):
    classCount = {}
    for instance in dataSet:
        currentclass = instance[-1]
        if currentclass in classCount:
            classCount[currentclass] += 1;
        else:
            classCount[currentclass] = 1;
    entropy = 0.0
    for key in classCount:
        prob = float(classCount[key])/len(dataSet)
        entropy -= prob*log(prob,2)
    return entropy



def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    #Last column contains the class label to which this entry belongs.
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeatureIndex = -1
    for i in range(numFeatures):
        featureValueList = [instance[i] for instance in dataSet]
        uniqueFeatureValue = set(featureValueList)
        newEntropy = 0.0
        for value in uniqueFeatureValue:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = float(len(subDataSet))/len(dataSet)
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if bestInfoGain < infoGain :
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex



def splitDataSet(dataSet, index, value):
    retDataSet = []
    for instance in dataSet:
        if instance[index]==value:
            reducedInstance = instance[:index]
            reducedInstance.extend(instance[index+1:])
            retDataSet.append(reducedInstance)
    return retDataSet



def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #sortedClassCount is a list of list. Entries of dictionary are converted to list and are
    #sorted in decreasing order according to the value of (key:value) pair.
    return sortedClassCount[0][0]



def createTree(dataSet, features):
    classList = [instance[-1] for instance in dataSet]
    #Stop when all classes are equal
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    #Stop when no more features  left, return majority
    if len(dataSet[0])==1:
        return majorityCount(classList)
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeature = features[bestFeatureIndex]
    del(features[bestFeatureIndex])
    myTree = {bestFeature: {}}
    featureValues = [instance[bestFeatureIndex] for instance in dataSet]
    uniqueFeatureValues = set(featureValues)
    for value in uniqueFeatureValues:
        #Creating a copy of list and passing the copy is necessary form proper functioning. This is because python passes list by reference, and we are doing dfs traversal, so features will be lost and wrong output.
        subfeatures = features[:]
        myTree[bestFeature][value] = createTree(splitDataSet(dataSet,bestFeatureIndex,value), subfeatures)
    return myTree



def createDataSet():
    myDataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    myFeatures = ['no surfacing', 'flippers']
    return myDataSet, myFeatures



def storeTree(myTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(myTree, fw)
    fw.close()



def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)



def classify(myTree, myFeatures, testVector):
    nodeStr = list(myTree)[0]
    nestedDict = myTree[nodeStr]
    featureIndex = myFeatures.index(nodeStr)
    for key in nestedDict.keys():
        if testVector[featureIndex]==key:
            if type(nestedDict[key]).__name__=='dict':
                classLabel = classify(nestedDict[key], myFeatures, testVector)
            else:
                classLabel = nestedDict[key]
    return classLabel
