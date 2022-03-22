from locale import currency
from random import random
from sqlite3 import DatabaseError
import numpy as np
import matplotlib.pyplot as plt
import math
import time


NOISE = 0
UNCLASSIFIED = False
def createdataset(filename,splitchar):
    # 创建数据库，输入数据
    fr = open(filename,"r")
    Dataset = []
    for line in fr.readlines():
        curline = line.split(splitchar)
        dataline = list(map(float,curline))
        Dataset.append(dataline)
    fr.close()

    return Dataset

def distance(a,b):
    # 计算欧氏距离
    return math.sqrt(np.power(a-b,2).sum())

def eps_calculate(a,b,eps):
    # 判断是否小于eps
    return distance(a,b)<eps

def region_judge(dataset,pointID,eps):
    # 判断是否在范围内,返回需要的点的ID
    columns = dataset.shape[1]
    Dataset_judged = []
    for i in range(columns):
        if eps_calculate(dataset[:,pointID],dataset[:,i],eps):
            Dataset_judged.append(i)
    return Dataset_judged




def DBSCAN(dataset,eps,MinPts):
    clusterID = 1
    cloumns = dataset.shape[1]
    clusterResult = [UNCLASSIFIED]*cloumns
    
    for pointid in range(cloumns):
        if clusterResult[pointid]==UNCLASSIFIED:
            Dataset_judged = region_judge(dataset,pointid,eps)
            # 确定某区域内存在可聚类点
            if len(Dataset_judged)>=MinPts :
                # 开始循环
                while len(Dataset_judged)>0:
                    # 从簇里面的第一个点开始
                    currentPoint = Dataset_judged[0]
                    Dataset_judged2 = region_judge(dataset,currentPoint,eps)
                    # 第一个点形成新的簇
                    if len(Dataset_judged2) >= MinPts:
                        # 循环新簇内第一个点
                        for i in range(len(Dataset_judged2)):
                            resultpoint = Dataset_judged2[i]
                            # 未识别的点进入原簇
                            if clusterResult[resultpoint] == UNCLASSIFIED:
                                Dataset_judged.append(resultpoint)
                                clusterResult[resultpoint]=clusterID
                            # 识别为NOISE的点失去进入原簇的价值
                            elif clusterResult[resultpoint]==NOISE:
                                clusterResult[resultpoint]=clusterID
                    # 删去原簇的第一个点
                    Dataset_judged = Dataset_judged[1:]
                # 循环结束，簇数加一
                clusterID = clusterID + 1
            else:
                clusterResult[pointid] = NOISE  
    return clusterID - 1,clusterResult


def iszero(a):
    i=0
    for j in a:
        if int(j)!=0:
            i +=1
    if i == 0:
        return False
    else:
        return True


def plotFeature(Dataset, cluster, clusterNum):
    matClusters = np.mat(cluster).transpose()
    fig = plt.figure()
    # fr=open('hotel2_DBSCAN.txt','r')
    # dataset = []
    # for line in fr.readlines():
    #     curline = line.split('\t')
    #     dataline = list(map(float,curline))
    #     dataset.append(dataline)
    # fr.close()
    # dataset = np.mat(dataset).transpose()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum+1):
        colorSytle = scatterColors[i % len(scatterColors)]
        print(np.nonzero(matClusters[:, 0].A == i)[0])
        subCluster = Dataset[:, np.nonzero(matClusters[:, 0].A == i)[0]]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=30)
        print(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], colorSytle)

  

def main():
    Dataset = createdataset('hotel2.txt', '\t')
    Dataset = np.mat(Dataset).transpose()
    clusterNum,cluster = DBSCAN(Dataset,0.01,5)
    print("簇数为 = ", clusterNum)
    curline = open('hotel2.txt').read().splitlines()
    out = open('hotel2_DBSCAN.txt','w')
    i=0
    for line in curline:
        out.write(line + '\t{0}\n'.format(cluster[i]))
        i = i+1
    out.close()
    
    plotFeature(Dataset,cluster,clusterNum)

if __name__ == '__main__':
    main()
    plt.show()

    



    







