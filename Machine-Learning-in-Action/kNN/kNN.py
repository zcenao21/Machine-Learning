from numpy import *
import matplotlib.pyplot as plt
import operator

# 读取文件到
def file2matrix(filename):
    fr=open(filename)
    arrayOfLines=fr.readlines()
    numberOfLines=len(arrayOfLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOfLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

# 数据归一化
def normData(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    line=dataSet.shape[0]
    nData=(dataSet-tile(minVals,(line,1)))/tile(ranges,(line,1))
    return  nData,ranges,minVals

# k近邻算法
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    sqDiffMat=(tile(inX,(dataSetSize,1))-dataSet)**2
    distances=(sqDiffMat.sum(axis=1))**0.5
    sortedDistances=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistances[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 测试代码
def datingClassTest():
    # 载入数据并显示
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    print(datingDataMat)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],20*array(datingLabels),20*array(datingLabels))

    # 数据归一化
    normMat,ranges,minVals=normData(datingDataMat)
    fig2=plt.figure()
    bx=fig2.add_subplot(111)
    bx.scatter(normMat[:,0],normMat[:,1],20*array(datingLabels),20*array(datingLabels))

    m=normMat.shape[0]
    hoRatio=0.1
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %d, the real answer is: %d'
              % (classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def showInFig(normMat,datingLabels):
    fig2=plt.figure()
    bx=fig2.add_subplot(111)
    bx.scatter(normMat[:,0],normMat[:,1],20*array(datingLabels),20*array(datingLabels))
    plt.show()

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("Frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=normData(datingDataMat)
    showInFig(normMat,datingLabels)
    inArr=array([ffMiles,percentTats,iceCream])
    classsifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person: ', resultList[classsifierResult-1])

