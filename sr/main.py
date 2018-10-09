#!/usr/bin/python
# -*- coding: utf-8 -*-

#---------------------------从文本中构建词条向量-------------------------
#1 要从文本中获取特征，需要先拆分文本，这里特征是指来自文本的词条，每个词
#条是字符的任意组合。词条可以理解为单词，当然也可以是非单词词条，比如URL
#IP地址或者其他任意字符串 
#  将文本拆分成词条向量后，将每一个文本片段表示为一个词条向量，值为1表示出现
#在文档中，值为0表示词条未出现


#导入numpy
from numpy import *

def loadDataSet():
#词条切分后的文档集合，列表每一行代表一个文档
    postingList=[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['my','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    #由人工标注的每篇文档的类标签
    classVec=[0,1,0,1,0,1]
    return postingList,classVec


#统计所有文档中出现的词条列表    
def createVocabList(dataSet):
    #新建一个存放词条的集合
    vocabSet=set([])
    #遍历文档集合中的每一篇文档
    for document in dataSet:
        #将文档列表转为集合的形式，保证每个词条的唯一性
        #然后与vocabSet取并集，向vocabSet中添加没有出现
        #的新的词条        
        vocabSet=vocabSet|set(document)
    #再将集合转化为列表，便于接下来的处理
    return list(vocabSet)

#根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量    
def setOfWords2Vec(vocabSet,inputSet):
    #新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec=[0]*len(vocabSet)
    #遍历文档中的每一个词条
    for word in inputSet:
        #如果词条在词条列表中出现
        if word in vocabSet:
            #通过列表获取当前word的索引(下标)
            #将词条向量中的对应下标的项由0改为1
            returnVec[vocabSet.index(word)]=1
        else: print('the word: %s is not in my vocabulary! '%'word')
    #返回inputet转化后的词条向量
    return returnVec

#训练算法，从词向量计算概率p(w0|ci)...及p(ci)
#@trainMatrix：由每篇文档的词条向量组成的文档矩阵
#@trainCategory:每篇文档的类标签组成的向量
def trainNB0(trainMatrix,trainCategory):
    #获取文档矩阵中文档的数目
    numTrainDocs=len(trainMatrix)
    #获取词条向量的长度
    numWords=len(trainMatrix[0])
    #所有文档中属于类1所占的比例p(c=1)
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #创建一个长度为词条向量等长的列表
    p0Num=zeros(numWords);p1Num=zeros(numWords)
    p0Denom=0.0;p1Denom=0.0
    #遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        #如果该词条向量对应的标签为1
        if trainCategory[i]==1:
            #统计所有类别为1的词条向量中各个词条出现的次数
            p1Num+=trainMatrix[i]
            #统计类别为1的词条向量中出现的所有词条的总数
            #即统计类1所有文档中出现单词的数目
            p1Denom+=sum(trainMatrix[i])
        else:
            #统计所有类别为0的词条向量中各个词条出现的次数
            p0Num+=trainMatrix[i]
            #统计类别为0的词条向量中出现的所有词条的总数
            #即统计类0所有文档中出现单词的数目
            p0Denom+=sum(trainMatrix[i])
    #利用NumPy数组计算p(wi|c1)
    p1Vect=p1Num/p1Denom  #为避免下溢出问题，后面会改为log()
    #利用NumPy数组计算p(wi|c0)
    p0Vect=p0Num/p0Denom  #为避免下溢出问题，后面会改为log()
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
#@vec2Classify:待测试分类的词条向量
#@p0Vec:类别0所有文档中各个词条出现的频数p(wi|c0)
#@p0Vec:类别1所有文档中各个词条出现的频数p(wi|c1)
#@pClass1:类别为1的文档占文档总数比例
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


#分类测试整体函数        
def testingNB():
    #由数据集获取文档矩阵和类标签向量
    listOPosts,listClasses=loadDataSet()
    #统计所有文档中出现的词条，存入词条列表
    myVocabList=createVocabList(listOPosts)
    #创建新的列表
    trainMat=[]
    for postinDoc in listOPosts:
        #将每篇文档利用words2Vec函数转为词条向量，存入文档矩阵中
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))\
    #将文档矩阵和类标签向量转为NumPy的数组形式，方便接下来的概率计算
    #调用训练函数，得到相应概率值
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    #测试文档
    testEntry=['love','my','dalmation']
    #将测试文档转为词条向量，并转为NumPy数组的形式
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    #利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    #第二个测试文档
    testEntry1=['stupid','garbage']
    #同样转为词条向量，并转为NumPy数组的形式
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classifyNB(thisDoc1,p0V,p1V,pAb))


def main():
    testingNB()
    pass


if __name__ == "__main__":
    main()
    pass

