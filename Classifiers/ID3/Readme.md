**Import ID3 algorithm.**
```
>>> import Decision_Tree_ID3_Classifier as dt
```

**Get default dataset and feature set.**
```
>>> myData, myFeatures = dt.createDataSet()
>>> myData
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
>>> myFeatures
['no surfacing', 'flippers']
```

**Create tree**
```
>>> tree = dt.createTree(myData, myFeatures)
>>> tree
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
>>> myData, myFeatures = dt.createDataSet()
```

**Run Classifier**
```
>>> dt.classify(tree, myFeatures, [1, 0])
'no'
>>> dt.classify(tree, myFeatures, [1, 1])
'yes'
>>> dt.classify(tree, myFeatures, [0, 1])
'no'
>>> dt.classify(tree, myFeatures, [0, 0])
'no'
```
