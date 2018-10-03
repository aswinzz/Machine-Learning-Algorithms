from numpy import *
import operator
from os import listdir
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndices = distances.argsort()
	classCount={}

	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOflines = len(fr.readlines())
	returnMat = zeros((numberOflines,3))
	label = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		lineFromLine = line.split('\t')
		returnMat[index,:] = lineFromLine[0:3]
		label.append(lineFromLine[3])
		index += 1
	return returnMat,label

def autonorm(dataset):
	minVals = dataset.min(axis = 0)
	maxVals = dataset.max(axis = 0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataset))
	n = dataset.shape[0]
	normDataset = dataset - tile(minVals,(n,1))
	normDataset = normDataset / tile(maxVals,(n,1))
	return normDataset,ranges,minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat,datingLabels = file2matrix("datingTestSet.txt")
	normMat,ranges,minVals = autonorm(datingDataMat)
	n = normMat.shape[0]
	numTestVecs = (int)(n*hoRatio)
	error = 0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:n,:],datingLabels[numTestVecs:n],3)
		print("The classifier come with result : %s ,the real answer is : %s" %(classifierResult,datingLabels[i]) )
		if classifierResult != datingLabels[i]:
			error += 1.0
	print("Error : %f" %(error/float(numTestVecs)) )

def classifyPerson():
	percentTats = float(input("Percentage Of time spent playing video games : "))
	ffMiles = float(input("Frequent flyer miles earned per year : "))
	iceCream = float(input("Liters of icecream consumed per year : "))
	datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
	normMat,ranges,minVals = autonorm(datingDataMat)
	inArr = array([ffMiles,percentTats,iceCream])
	classifierResult = classify0( (inArr-minVals)/ranges ,normMat,datingLabels,3 )
	print("Result : " , classifierResult )
