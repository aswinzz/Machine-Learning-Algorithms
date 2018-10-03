## Dataset Information
The data  collected is in a text file called **datingTestSet.txt**.It has 1,000 entries. A new sample is on each line, and
we have recorded the following **features**:    
■ Number of frequent flyer miles earned per year     
■ Percentage of time spent playing video games        
■ Liters of ice cream consumed per week         
**Labels**:   
■ People not liked     
■ People liked in small doses          
■ People liked in large doses       

## Testing KNN Algorithm  

90% data is used to train the algorithm and 10% of it is used to test it.

In [2]: datingClassTest()  

The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : didntLike ,the real answer is : didntLike
The classifier come with result : largeDoses ,the real answer is : smallDoses
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : largeDoses ,the real answer is : largeDoses
The classifier come with result : smallDoses ,the real answer is : smallDoses
The classifier come with result : didntLike ,the real answer is : didntLike
...
Error : 0.050000

## Classify Person

In[3]: classifyPerson()

Percentage Of time spent playing video games : 8

Frequent flyer miles earned per year : 40000

Liters of icecream consumed per year : 0.95

Result :  in large doses
