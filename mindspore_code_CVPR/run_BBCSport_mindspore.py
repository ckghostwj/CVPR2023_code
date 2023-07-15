#!/usr/bin/env python
# coding: utf-8

# This is the source code using Mindspore for the paper:
# Wen J, Liu C, Xu G, et al. 
# Highly Confident Local Structure Based Consensus Graph Learning for Incomplete Multi-View Clustering
# in: CVPR. 2023: 15712-15721.

# In[1]:


import numpy as np
import math
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from scipy.sparse import spdiags
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
import scipy.io as scio
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.numpy as np2


# In[3]:


def HCLS_CGL(Skiv,Wiv,XWiv,F_ini,num_clust,lambda2,lambda3,para_r,max_iter):
    num_view=len(Skiv)
    F=F_ini.copy()
    alpha=np.ones(num_view)
    alpha_r=alpha**para_r
    Z_linshi=0
    W_linshi=0
    Wiv2=[]
    for iv in range(num_view):
        Z_linshi=Z_linshi+Skiv[iv]*(XWiv[iv])
        W_linshi=W_linshi+XWiv[iv]
        Wiv2.append(Wiv[iv]*(Wiv[iv]))
    Wiv2=np.array(Wiv2)
    eps = 1e-8
    Z=Z_linshi/(W_linshi+eps)
    Z[np.isnan(Z)] = 0
    Z[np.isinf(Z)] = 1
    obj=[]
    for it in range(max_iter):
        linshi_fenzi=0
        linshi_fenmu=0
        for iv in range(num_view):
            linshi_fenzi=linshi_fenzi+alpha_r[iv]*(Skiv[iv])*(Wiv2[iv])
            linshi_fenmu=linshi_fenmu+alpha_r[iv]*(Wiv2[iv])
        linshi_P=EuDist2(F,F,0)
        linshi_P=Tensor(linshi_P,mstype.float64)
        linshi_P=linshi_P-np2.diag(np2.diag(linshi_P))
        linshi_P=linshi_P.asnumpy()
        linshi_Z=(linshi_fenzi-0.25*lambda3*linshi_P)/(linshi_fenmu+lambda2)
        Z=np.zeros_like(linshi_Z)
        for i in range(Z.shape[1]):
            linshi_c = np.arange(linshi_Z.shape[0])
            linshi_c=np.delete(linshi_c,i, axis=0)
            x_temp,ft_temp=EProjSimplex_new(linshi_Z[linshi_c,i])
            for j in range(len(linshi_c)):
                Z[linshi_c[j]][i]=x_temp[j]
        linshiZ=(Z+Z.T)*0.5
        linshiZ=Tensor(linshiZ,mstype.float64)
        LapZ=np2.diag(np2.sum(linshiZ,axis=0))-linshiZ
        LapZ=LapZ.asnumpy()
        F,_,_=eig1(LapZ,num_clust,0)
        Rec_error=[]
        for iv in range(num_view):
            error = np.linalg.norm((Z - Skiv[iv]) * Wiv[iv], 'fro')**2
            Rec_error.append(error)
        Rec_error=np.array(Rec_error)
        linshi_H = np.power(Rec_error, 1 / (1 - para_r))
        alpha =linshi_H / np.sum(linshi_H)
        alpha_r=alpha**para_r
        linshiZ=(Z+Z.T)*0.5
        linshiZ=Tensor(linshiZ,mstype.float64)
        LapZ=np2.diag(np2.sum(linshiZ,axis=0))-linshiZ
        LapZ=LapZ.asnumpy()
        temp=alpha_r.dot(Rec_error.T)+lambda3*np.trace(F.T.dot(LapZ).dot(F)) + lambda2 * (np.linalg.norm(Z, 'fro') ** 2)
        obj.append(temp)
        if it >2 and abs(obj[it]-obj[it-1])<1e-6:
            break
    F=np.array(F)
    Z=np.array(Z)
    obj=np.array(obj) 
    return F,Z,obj


# In[4]:


def readsparse(Xs):
    numRow, numCol = Xs[0, :2]
    Xf = np.zeros((numRow, numCol))
    for i in range(1,Xs.shape[0]):
        row_idx, col_idx, value = Xs[i]
        Xf[row_idx-1, col_idx-1] = value
    return Xf


# In[5]:


def solveF(Z,numOfClasses):
    numOfViews=len(Z)
    sumLZ = 0
    for i in range(numOfViews):
        W=0.5*(Z[i]+Z[i].T)
        temp=np.sum(W,axis=1)
        temp2=np.diag(temp)
        M=temp2-W
        M[np.isnan(M)]=0
        M[np.isinf(M)]=1e5
        sumLZ = sumLZ+M
    try:
        D,V=np.linalg.eig(sumLZ)
    except Exception as e:
        if e.__class__.__name__ == 'LinAlgError' and str(e).startswith('Eigenvalues did not converge'):
            D, V = np.linalg.eig(sumLZ)
        else:
            raise e
    # 调用 numpy 中的 argsort 函数获取排序后的索引值
    ind = np.argsort(D)
    # 使用索引值对排序后的对角线元素进行重新排序
    D = D[ind]
    V = np.concatenate([V[:, i:i+1] for i in ind], axis=1)
    ind = np.argsort(D)
    D_sort=D[ind]
    i=0
    ind2=[]
    for i in range(len(D_sort)):
        if D_sort[i]>1e-6:
            ind2.append(i)
    if len(ind2)>numOfClasses:
        F=V[:,ind2[0:numOfClasses]]  #matlab从1计数，python从0记数
    else:
        F=V[:,ind[0:numOfClasses]]
    return F


# In[6]:


# In[7]:


def NormalizeFea(fea,row):
    if not 'row' in locals():
        row=1
    if row:
        nSmp=fea.shape[0]
        # fea: 特征矩阵
        feaNorm = np.sum(fea ** 2, axis=1)  # 计算每个行向量的 L2 范数的平方
        feaNorm = np.maximum(feaNorm, 1e-14)  # 避免除以零错误
        # 计算每个样本特征的L2范式
        feaNorm = np.linalg.norm(fea, axis=1)
        # 将计算结果转化成对角阵
        feaNorm=Tensor(feaNorm,mstype.float64)
        D =np2.diag(1.0/np2.sqrt(feaNorm))
        D=D.asnumpy()
        # 左乘对角阵，完成矩阵行归一化
        fea = D.dot(fea)
    else:
        nSmp=fea.shape[1]
        # fea: 特征矩阵
        feaNorm = np.sum(fea ** 2, axis=0)  # 计算每个行向量的 L2 范数的平方
        feaNorm = np.maximum(feaNorm, 1e-14)  # 避免除以零错误
        # 计算每个样本特征的L2范式
        # 计算每一行的范数的平方 
        feaNorm = np.linalg.norm(fea, axis=0)**2
        # 构造对角矩阵，取倒数的平方根为对角元素
        temp=Tensor(feaNorm,mstype.float64)
        temp=np2.sqrt(temp)
        temp=temp.asnumpy()
        D = spdiags(1/temp, 0, len(feaNorm), len(feaNorm))
        # 将计算结果转化成对角阵
        feaNorm=Tensor(feaNorm,mstype.float64)
        D = np2.diag(1.0/np2.sqrt(feaNorm))
        D=D.asnumpy()
        # 左乘对角阵，完成矩阵行归一化
        fea = fea.dot(D)
    return fea


# In[8]:


def EuDist2(fea_a, fea_b=None, bSqrt=1,mark=0):
    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1)
        ab = np.dot(fea_a, fea_a.T)
        if issparse(aa):
            aa = aa.toarray()
        
        D = np.add.outer(aa, aa) - 2 * ab
        D[D < 0] = 0
        
        if bSqrt:
            D=Tensor(D,mstype.float64)
            D = np2.sqrt(D)
            D=D.asnumpy()
        D = np.maximum(D, D.T)
    else:
        aa = np.sum(fea_a * fea_a, axis=1)
        bb = np.sum(fea_b * fea_b, axis=1)
        ab = np.dot(fea_a, fea_b.T)
        if issparse(aa):
            aa = aa.toarray()
            bb = bb.toarray()
        
        D = np.add.outer(aa, bb) - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D=Tensor(D,mstype.float64)
            D = np2.sqrt(D)
            D=D.numpy()
    return D


# In[9]:


def EProjSimplex_new(v,k=1):
    ft=1
    n=len(v)
    v0 = v - np.mean(v) + k/n
    vmin=np.min(v0)
    if vmin<0:
        f=1
        lambda_m=0
        while abs(f)>10**-10:
            v1=v0-lambda_m
            posidx=v1>0
            npos=np.sum(posidx)
            g=-npos
            f=np.sum(v1[posidx])-k
            lambda_m=lambda_m-f/g
            ft=ft+1
            if ft>100:
                x=np.max(v1,0)
                break
        x=[]
        for i in range(len(v1)):
            if v1[i]<0:
                x.append(0)
            else:
                x.append(v1[i])
        x=np.array(x)
    else:
        x=v0
    return x,ft


# In[10]:


def eig1(A,c=None,isMax=1,isSym=1):
    if c==None:
        c = A.shape[0]
    if isSym == 1:
        A = np.maximum(A, A.T)
    try:
        d, v = np.linalg.eig(A)
    except np.linalg.LinAlgError as e:
        if 'No convergence' in str(e):
            d, v = np.linalg.eig(A, np.eye(A.shape[0]))
        else:
            raise e
    d=Tensor(d,mstype.float64)
    d = np2.diag(np2.diag(d))
    d=d.asnumpy()
    if isMax == 0:
        idx = np.argsort(d)
    else:
        idx = np.argsort(-d)
    idx1=[]
    for i in range(c):
        idx1.append(idx[i])
    idx1=np.array(idx1)
    eigval = d[idx1]
    eigvec = v[:, idx1]
    eigval_full = d[idx]
    return eigvec,eigval,eigval_full



# In[12]:


def constructW(fea, options=None):
    bSpeed = 1

    if options is None:
        options = {}

    if 'Metric' in options:
        print("Warning: This function has been changed and the Metric is no longer supported")

    if 'bNormalized' not in options:
        options['bNormalized'] = 0
    
####################################
    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'

    if options['NeighborMode'].lower() == 'knn':
        if 'k' not in options:
            options['k'] = 5

    else:
       raise ValueError("NeighborMode does not exist!")
#######################################
    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'

    bBinary = False
    bCosine = False
    
    
    if options['WeightMode'].lower() == 'binary':
        bBinary = True
    elif options['WeightMode'].lower() == 'heatkernel':
        if 't' not in options:
            nSmp = fea.shape[0]
            if nSmp > 3000:
                D = EuDist2(fea[np.random.choice(nSmp, 3000)], )
            else:
                D = EuDist2(fea)
                #print(D)
            options['t'] = np.mean(np.mean(D))
    elif options['WeightMode'].lower() == 'cosine':
        bCosine = True
    else:
       raise ValueError("WeightMode does not exist!")
#######################################
    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = False
#######################################
    if 'gnd' in options:
        nSmp = len(options['gnd'])
    else:
        nSmp = fea.shape[0]

    maxM = 62500000  # 500MB
    BlockSize = maxM // (nSmp * 3)
    #print(BlockSize)
    
    if bCosine and not options['bNormalized']:
        Normfea = NormalizeFea(fea)

    if options['NeighborMode'] == 'KNN' and options['k'] > 0:
        if not(bCosine and options['bNormalized']):
            G = np.zeros((nSmp*(options['k']+1),3))
            for i in range(1, math.ceil(nSmp/BlockSize)+1):
                if i == math.ceil(nSmp/BlockSize):
                    smpIdx = np.arange((i-1)*BlockSize,nSmp)
                    dist = EuDist2(fea[smpIdx,:], fea,0)
                    dist=np.array(dist)
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = dump.copy()
                        for j in range(0, options['k']+1):
                            dump[:,j]=np.min(dist,axis=1)
                            idx[:,j]=np.argmin(dist,axis=1)
                            temp = (idx[:, j]) * nSmpNow + np.arange(1, nSmpNow+1)
                            temp=np.array(temp,np.int32)
                            temp=temp-1
                            for m in range(len(temp)):
                                m1=int(temp[m]%dist.shape[0])
                                m2=int(temp[m]/dist.shape[0])
                                dist[m1][m2]=1e100
                    else:
                        idx = np.argsort(dist, axis=1)[:, :options['k']+1]
                        dump = dist[np.arange(len(smpIdx))[:,None], idx]
                    if not bBinary:
                        if bCosine:
                            dist = np.dot(Normfea[smpIdx,:], Normfea.T)
                            linidx = np.arange(len(idx))[:,None]
                            dump = dist[linidx, idx]
                        else:
                            dump = np.exp(-dump/(2*options['t']**2))
                    G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1),0] = np.tile(smpIdx,options['k']+1).flatten()
                    G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1),1] = idx.T.flatten()
                    if not bBinary:
                        G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1),2] = dump.T.flatten()
                    else:
                        G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1),2] = 1
                else:
                    smpIdx = np.arange((i-1)*BlockSize,i*BlockSize)
                    dist = EuDist2(fea[smpIdx,:], fea)
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = dump.copy()
                        for j in range(0, options['k']+1):
                            dump[:,j],idx[:,j] = np.min(dist, axis=1), np.argmin(dist, axis=1)
                            temp = (idx[:,j])*nSmpNow + np.arange(1, nSmpNow+1)
                            #print(temp)
                            temp=np.array(temp,np.int32)
                            temp=temp-1
                            np.put(dist, temp, 1e100)
                    else:
                        idx = np.argsort(dist, axis=1)[:, :options['k']+1]
                        dump = dist[np.arange(len(smpIdx))[:,None], idx]

                    if not bBinary:
                        if bCosine:
                            dist = np.dot(Normfea[smpIdx,:], Normfea.T)
                            linidx = np.arange(len(idx))[:,None]
                            dump = dist[linidx, idx]
                        else:
                            dump = np.exp(-dump/(2*options['t']**2))
                        
                    G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1),0] = np.tile(smpIdx, (options['k']+1,1)).T.flatten()
                    G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1),1] = idx.flatten()
                    if not bBinary:
                        G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1),2] = dump.flatten()
                    else:
                        G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1),2] = 1
            W = csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))
        else:
            G = np.zeros((nSmp*(options['k']+1), 3))
            for i in range(1, int(np.ceil(nSmp/BlockSize))+1):
                if i == int(np.ceil(nSmp/BlockSize)):
                    smpIdx = list(range((i-1)*BlockSize+1, nSmp+1))
                    dist = fea[smpIdx, :].dot(fea.T)
                    dist = np.array(dist.todense())

                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k']+1))
                        idx = dump.copy()
                        for j in range(options['k']+1):
                            idx[:,j] = np.argmax(dist, axis=1)
                            dump[:,j] = dist[np.arange(nSmpNow), idx[:,j]].squeeze()
                            temp = (idx[:,j])*nSmpNow + np.arange(1, nSmpNow+1)
                            dist.flat[temp-1] = 0
                    else:
                        idx = np.argsort(-dist, axis=1)[:, :options['k']+1]
                        dump = -np.sort(-dist, axis=1)[:, :options['k']+1]

                    G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1), 0] = np.repeat(
                        smpIdx, (options['k']+1))
                    G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1), 1] = idx.flatten()
                    G[(i-1)*BlockSize*(options['k']+1):nSmp*(options['k']+1), 2] = dump.flatten()
                else:
                    smpIdx = list(range((i-1)*BlockSize+1, i*BlockSize+1))
                    dist = fea[smpIdx, :].dot(fea.T)
                    dist = np.array(dist.todense())
                    
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k']+1))
                        idx = dump.copy()
                        for j in range(options['k']+1):
                            idx[:,j] = np.argmax(dist, axis=1)
                            dump[:,j] = dist[np.arange(nSmpNow), idx[:,j]].squeeze()
                            temp = (idx[:,j]-1)*nSmpNow + np.arange(1, nSmpNow+1)
                            dist.flat[temp-1] = 0
                    else:
                        idx = np.argsort(-dist, axis=1)[:, :options['k']+1]
                        dump = -np.sort(-dist, axis=1)[:, :options['k']+1]

                    G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1), 0] = np.repeat(
                        smpIdx, (options['k']+1))
                    G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1), 1] = idx.flatten()
                    G[(i-1)*BlockSize*(options['k']+1):i*BlockSize*(options['k']+1), 2] = dump.flatten()

            W = csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))

        if bBinary:
            W = np.array(W.todense(), dtype=bool)
            # 索引布尔型数组中 True 所在位置，并将这些位置上的元素赋值为 1
            W[W == True] = 1
            W=coo_matrix(W)
        if 'bSemiSupervised' in options and options['bSemiSupervised']:
            tmpgnd = options['gnd'][options['semiSplit']]
            
            Label = np.unique(tmpgnd)
            nLabel = len(Label)
            G = np.zeros((np.sum(options['semiSplit']), np.sum(options['semiSplit'])))
            for idx in range(nLabel):
                classIdx = tmpgnd == Label[idx]
                G[classIdx, classIdx] = 1
            Wsup = sparse.csr_matrix(G)
            if 'SameCategoryWeight' not in options:
                options['SameCategoryWeight'] = 1
            W[options['semiSplit'], options['semiSplit']] = (Wsup > 0) * options['SameCategoryWeight']
        if not options['bSelfConnected']:
            c = W.todense()
            c=np.array(c)
            for n in range(c.shape[0]):
                c[n][n]=0
            # 取出对角线上的元素组成一维数组，再构造一个以此数组为对角线的矩阵
            # 将 D 用 np.multiply 函数乘以 -1，再与 W 相加就可以得到目标稀疏矩阵
            W=coo_matrix(c)

        if 'bTrueKNN' in options and options['bTrueKNN']:
            pass
        else:
            a1=W.todense()
            a1=np.array(a1)
            a2=W.T.todense()
            a2=np.array(a2)
            a3 = np.maximum(a1,a2)
            W=coo_matrix(a3)

        return W
    weight_mode = options['WeightMode'].lower()
    if weight_mode == 'binary':
        raise ValueError('Binary weight can not be used for complete graph!')
    elif weight_mode == 'heatkernel':
        W = EuDist2(fea)
        W = np.exp(-W / (2 * options['t'] ** 2))
    elif weight_mode == 'cosine':
        
        W = Normfea.dot(Normfea.T)
    else:
        raise ValueError('WeightMode does not exist!')
    if not options['bSelfConnected']:
        np.fill_diagonal(W, 0)
    #W = np.maximum(W, W.T)
    a1=W.todense()
    a1=np.array(a1)
    a2=W.T.todense()
    a2=np.array(a2)
    a3 = np.maximum(a1,a2)
    W=coo_matrix(a3)
    return W


# In[13]:


def compute_nmi(T, H):
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)

    # 计算每个类别中的点数
    D = np.zeros(num_class)
    for j in range(num_class):
        index_class = (T == classes[j])
        D[j] = sum(index_class)

    # 计算互信息和A矩阵
    mi = 0
    A = np.zeros((num_clust, num_class))
    avgent = 0
    for i in range(num_clust):
        # 在聚类' i '中的点数
        index_clust = (H == clusters[i])
        B_i = sum(index_clust)
        B = np.zeros(num_clust)
        B[i] = B_i
        for j in range(num_class):
            index_class = (T == classes[j])
            # 计算属于类'j'的点数，在聚类'i'中结束
            A[i][j] = sum(index_class & index_clust)
            if A[i][j] != 0:
                miarr_ij = A[i][j]/N * np.log2(N*A[i][j]/(B_i*D[j]))
                # 平均熵计算
                avgent -= (B_i/N) * (A[i][j]/B_i) * np.log2(A[i][j]/B_i)
            else:
                miarr_ij = 0
            mi += miarr_ij

    # 计算类别熵
    class_ent = 0
    class_ent = sum(D/N * np.log2(N/D))

    # 计算聚类熵
    clust_ent = 0
    clust_ent = sum(B/N * np.log2(N/B))

    # 计算归一化互信息
    if (clust_ent + class_ent) == 0:
        nmi = 0
    else:
        nmi = 2 * mi / (clust_ent + class_ent)

    return A, nmi, avgent


# In[14]:


def compute_f(T, H):
    if len(T) != len(H):  # 检查T和H长度是否相等，如果不相等则输出他们的size并直接结束函数
        print("size(T)=" + str(len(T)) + ", size(H)=" + str(len(H)))
        return

    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    
    for n in range(N):
        Tn = (T[n+1:] == T[n])  # 找到与当前T(n)相等的元素，并记录为逻辑值
        Hn = (H[n+1:] == H[n])  # 找到与当前H(n)相等的元素，并记录为逻辑值
        numT += sum(Tn)  # 记录在未来的所有操作中有多少个true值
        numH += sum(Hn)  # 记录在未来的所有操作中有多少个true值
        numI += sum(Tn * Hn)  # 记录在未来的所有操作中有多少个同时为true的情况
        
    p = 1
    r = 1
    f = 1
    if numH > 0:
        p = numI / numH  # 如果numH大于0，计算精确度p
    if numT > 0:
        r = numI / numT  # 如果numT大于0，计算召回率r
    if (p+r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)  # 计算F值
    return f, p, r


# In[15]:


def cnormalize_inplace(X, p=2):
    """
    CNORMALIZE_INPLACE normalizes columns.
    This is an inplace version of CNORMALIZE.

    :param X: input matrix
    :param p: the norm to use for normalization
    :return: normalized matrix and column norms (optional)
    """
    N = X.shape[1]
    if not p:
        p = 2

    # initialize output array if norm is requested
    #python没有nargout对应，返回值数量固定
    Xnorm = np.zeros((1, N))

    # loop through each column
    for iN in range(N):
        if p == np.Inf:
            cnorm = np.max(np.abs(X[:, iN]))
        else:
            cnorm = np.sum(np.abs(X[:, iN]) ** p) ** (1/p)
        X[:, iN] = X[:, iN] / (cnorm + np.finfo(float).eps)
        Xnorm[0,iN] = cnorm

    return X, Xnorm


# In[16]:


def cnormalize(X, p=2):
    if p == None:
        p = 2

    eps = np.finfo(float).eps
    if p == np.inf:
        Xnorm = np.max(abs(X), axis=0)
    else:
        Xnorm = np.sum(np.abs(X) ** p, axis=0) ** (1 / p)

    Y = X / (Xnorm + eps)

    return Y, Xnorm


# In[17]:

def hmreduce(A,CH,RH,LC,LR,SLC,SLR):
    A=np.delete(A,0,axis=0)
    A=np.delete(A,0,axis=1)
    CH=np.delete(CH,0)
    RH=np.delete(RH,0)
    LC=np.delete(LC,0)
    LR=np.delete(LR,0)
    SLC=np.array(SLC,np.int32)
    SLC=SLC-1
    SLR=np.array(SLR,np.int32)
    SLR=SLR-1
    n=A.shape[0]
    coveredRows=LR==0
    coveredCols=LC!=0
    r = np.where(~coveredRows)[0]
    c = np.where(~coveredCols)[0]
    m=1e100
    for i in range(len(r)):
        for j in range(len(c)):
            if m>A[r[i]][c[j]]:
                m=A[r[i]][c[j]]
    for i in range(len(r)):
        for j in range(len(c)):
            A[r[i]][c[j]]=A[r[i]][c[j]]-m
    for j in c:
        for i in  SLR:
            if (A[i,j]==0):
                if (RH[i]==0):
                    RH[i]=RH[n]
                    RH[n]=i+1
                    CH[i]=j+1
                row=A[i,:]
                colsInList=-np.extract(row<0, row)
                if (len(colsInList)==0):
                    l=n+1
                else:
                    l=colsInList[(row[colsInList]==0)-1][0]
                A[i,l-1]=-(j+1)
    r = np.where(coveredRows)[0]
    c = np.where(coveredCols)[0]
    i=[]
    j=[]
    for m1 in range(len(r)):
        for n1 in range(len(c)):
            if A[r[m1],c[n1]]<=0:
                i.append(m1)
                j.append(n1)
    i=r[i]
    j=c[j]
    for k in range(len(i)):
        lj=np.where(A[i[k],:] == -j[k])[0]
        A[i[k],lj]=A[i[k],j[k]]
        A[i[k],j[k]]=0
    for i in range(len(r)):
        for j in range(len(c)):
            A[r[i]][c[j]]=A[r[i]][c[j]]+m
    temp1=A.copy()
    temp2=CH.copy()
    temp3=RH.copy()
    A=np.zeros((A.shape[0]+1,A.shape[1]+1))
    CH=np.zeros(len(CH)+1)
    RH=np.zeros(len(RH)+1)
    for i in range(1,A.shape[0]):
        for j in range(1,A.shape[1]):
            A[i,j]=temp1[i-1,j-1]
    for i in range(1,len(CH)):
        CH[i]=temp2[i-1]
    for i in range(1,len(RH)):
        RH[i]=temp3[i-1] 
    A=np.array(A,np.int32)
    CH=np.array(CH,np.int32)
    RH=np.array(RH,np.int32)
    return A,CH,RH

def hmflip(A,C,LC,LR,U,l,r):
    count=0
    n=A.shape[0]
    while 1:
        C[l]=r
        count=count+1
        if count==100:
            break
        m=[]
        for i in range(1,A.shape[1]):
            if A[r,i]==-l:
                m.append(i)
        # print("m=",m)
        A[r,m]=A[r,l]
        A[r,l]=0
        if LR[r]<0:
            U[n]=U[r]
            U[r]=0
            return A,C,U
        else:
            l=LR[r]
            A[r,l]=A[r,n]
            A[r,n]=-l
            r=LC[l]


def hminiass(A):
    n, np1 = A.shape
    temp=A.copy()
    A=np.zeros((n+1,np1+1))
    for i in range(1,n+1):
        for j in range(1,np1+1):
            A[i,j]=temp[i-1,j-1]
    A=np.array(A,np.int32)
    C = np.zeros(n+1)
    C=np.array(C,np.int32)
    LZ = np.zeros(n+1)
    LZ=np.array(LZ,np.int32)
    NZ = np.zeros(n+1)
    NZ=np.array(NZ,np.int32)
    for i in range(1,n+1):
        lj=n+1
        j=-A[i,lj]
        while C[j]!=0:
            lj=j
            j=-A[i,lj]
            if j==0:
                break
        if j!=0:
            C[j]=i
            A[i,lj]=A[i,j]
            NZ[i]=-A[i,j]
            LZ[i]=lj
            A[i,j]=0
        else:
            lj=n+1
            j=-A[i,lj]
            while j!=0:
                r=C[j]
                r=int(r)
                lm=LZ[r]
                m=NZ[r]
                while m!=0:
                    if C[m]==0:
                        break
                    lm=m
                    m=-A[r,lm]
                if m==0:
                    lj=j
                    j=-A[i,lj]
                else:
                    A[r,lm]=-j
                    A[r,j]=A[r,m]
                    NZ[r]=-A[r,m]
                    LZ[r]=j
                    A[r,m]=0
                    C[m]=r
                    A[i,lj]=A[i,j]
                    NZ[i]=-A[i,j]
                    LZ[i]=lj
                    A[i,j]=0
                    C[j]=i
                    break
    
    
    C=np.delete(C,0)
    A=np.delete(A,0, axis=1)
    A=np.delete(A,0, axis=0)
    r=np.zeros(n)
    rows=[]
    for i in range(len(C)):
        if C[i]!=0:
            rows.append(C[i])
    rows=np.array(rows,np.int32)
    rows = rows[::-1]
    for i in range(len(rows)):
        r[rows[i]-1]=rows[i]
    empty=[]
    for i in range(len(r)):
        if r[i]==0:
            empty.append(i+1)
    U = np.zeros(n+1)
    if len(empty)==0:
            U[n]=0
    else:
        U[n]=empty[0]
        for i in range(len(empty)-1):
            empty[i]=empty[i+1]
        empty[len(empty)-1]=0
    return A,C,U



def hminired(A):
    m, n = np.shape(A)
    colMin = np.min(A, axis=0)
    A = A - colMin
    rowMin = np.min(A, axis=1)
    rowMin = rowMin.reshape((-1, 1)) #转换为列向量
    A = A - rowMin
    i=[]
    j=[]
    for m1 in range(A.shape[0]):
        for n1 in range(A.shape[1]):
            if A[n1][m1]==0:
                i.append(n1)
                j.append(m1)
    i=np.array(i,np.int32)
    j=np.array(j,np.int32)
    zeros_col = np.zeros((A.shape[0], 1)) # 创建值全为0且和arr行数一样的新列
    A = np.column_stack((A, zeros_col)) # 在 arr 的右侧加入新列
    for k in range(n):
        cols=[]
        for m in range(len(i)):
            if k==i[m]:
                cols.append(j[m])
        cols=np.array(cols)
        cols=np.reshape(cols,(len(cols),1))
        cols=cols.T
        if len(cols[0])==0:
            A[k,n]=0
        else:
            A[k,n]=-(cols[0][0]+1)
            for i1 in range(len(cols[0])-1):
                A[k,cols[0][i1]]=-(cols[0][i1+1]+1)
            A[k,cols[0][len(cols[0])-1]]=0
    return A

def hungarian(A):
    m,n = np.shape(A)
    if m != n:
        raise ValueError('Cost matrix must be square!')
    orig=A.copy()
    A=hminired(A)
    A,C,U=hminiass(A)
    temp1=A.copy()
    temp2=C.copy()
    temp3=U.copy()
    A=np.zeros((A.shape[0]+1,A.shape[1]+1))
    C=np.zeros(len(C)+1)
    U=np.zeros(len(U)+1)
    for i in range(1,A.shape[0]):
        for j in range(1,A.shape[1]):
            A[i,j]=temp1[i-1,j-1]
    for i in range(1,len(C)):
        C[i]=temp2[i-1]
    for i in range(1,len(U)):
        U[i]=temp3[i-1]
    A=np.array(A,np.int32)
    C=np.array(C,np.int32)
    U=np.array(U,np.int32)
    while (U[n+1]!= 0):
        LR = np.zeros(n+1)
        LC=np.zeros(n+1)
        CH=np.zeros(n+1)
        RH = np.concatenate((np.zeros(n+1), [-1]))
        RH=np.array(RH,np.int32)
        LR=np.array(LR,np.int32)
        LC=np.array(LC,np.int32)
        CH=np.array(CH,np.int32)
        SLC=[]
        r=U[n+1]
        r=int(r)
        LR[r]=-1
        SLR=[]
        SLR.append(r)
        while 1:
            if A[r,n+1]!=0:
                l=-A[r,n+1]
                if A[r,l]!=0 and RH[r]==0:
                    RH[r]=RH[n+1]
                    RH[n+1]=r
                    CH[r]=-A[r,l]
            else:
                if RH[n+1]<=0:
                    A,CH,RH=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
                r=RH[n+1]
                l=CH[r]
                CH[r]=-A[r,l]
                if A[r,l]==0:
                    RH[n+1]=RH[r]
                    RH[r]=0
            while LC[l]!=0:
                if RH[r]==0:
                    if RH[n+1]<=0:
                        A,CH,RH=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
                    r=RH[n+1]
                l=CH[r]
                CH[r]=-A[r,l]
                if A[r,l]==0:
                    RH[n+1]=RH[r]
                    RH[r]=0
            if C[l]==0:
                A,C,U=hmflip(A,C,LC,LR,U,l,r)
                break
            else:
                LC[l]=r
                SLC.append(l)
                r=C[l]
                LR[r]=l
                SLR.append(r)
    C=np.array(C,np.int32)
    C=np.delete(C,0)
    C_sparse = csc_matrix((np.ones_like(C), (C-1, np.arange(orig.shape[1]))))
    # 按照稀疏矩阵找到non-zero index，并根据这些index直接从`orig`数组中选取值并求和
    T=0
    rows, cols = C_sparse.nonzero()
    for i in range(len(rows)):
        T=T+orig[cols[i]][rows[i]]
    return C,T

def MutualInfo(L1,L2):
    L1 = np.squeeze(L1)
    L2 = np.squeeze(L2)
    if L1.shape != L2.shape:
        raise ValueError("Size of L1 must be equal to size of L2.")
    L1 = L1 - np.min(L1) + 1  
    L2 = L2 - np.min(L2) + 1   
    nClass = np.max([np.max(L1), np.max(L2)])
    nClass=int(nClass)
    G = np.zeros((nClass, nClass))
    eps=1e-20
    for i in range(nClass):
        for j in range(nClass):
            G[i,j] = np.sum((L1 == i+1) & (L2 == j+1)) + eps
    sumG = np.sum(G)
    P1 = G.sum(axis=1)
    P1 = P1/sumG
    P2 = G.sum(axis=0)
    P2 = P2/sumG
    H1 = np.sum(-P1*np.log2(P1))
    H2 = np.sum(-P2*np.log2(P2))
    # 计算P12和PPP，并计算MI
    P12 = G/sumG
    PPP = P12 / np.outer(P1,P2)
    PPP[np.abs(PPP) < 1e-12] = 1
    MI = np.sum(P12*np.log2(PPP))
    # 计算归一化的MI
    MIhat = MI / max(H1,H2)
    MIhat = np.real(MIhat)
    return MIhat

def bestMap(L1, L2):
    L1 = np.squeeze(L1)
    L2 = np.squeeze(L2)
    if L1.shape != L2.shape:
        raise ValueError("Size of L1 must be equal to size of L2.")
    L1 = L1 - np.min(L1) + 1   
    L2 = L2 - np.min(L2) + 1   
    nClass = np.max([np.max(L1), np.max(L2)])
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum(np.logical_and(L1 == i+1, L2 == j+1))
    [c,t] = hungarian(-G)
    newL2 = np.zeros((len(L2), 1))
    for i in range(len(L2)):
        newL2[i]=c[L2[i]-1]
    return newL2,c

def ClusteringMeasure(Y, predY):
    if Y.shape[1] != 1:
        Y = np.transpose(Y) 
    if predY.shape[1] != 1:
        predY = np.transpose(predY)
    n = len(Y)
    
    uY = np.unique(Y)
    nclass = len(uY)
    Y0 = np.zeros(n)
    if nclass != np.max(Y):
        for i in range(nclass):
            Y0[Y == uY[i]] = (i+1)
        Y = Y0.astype(int)
        
    uY = np.unique(predY)
    nclass = len(uY)
    predY0 = np.zeros(n)
    if nclass != np.max(predY):
        for i in range(nclass):
            predY0[predY == uY[i]] = (i+1)
        predY = predY0.astype(int)
    
    predLidx = np.unique(predY)
    pred_classnum = len(predLidx)
    # 计算purity
    correnum = 0
    for ci in range(pred_classnum):
        incluster = Y[predY == predLidx[ci]]
        inclunub = np.histogram(incluster, bins=range(1, max(incluster)+2))[0]
        if len(inclunub) == 0:
            inclunub = [0]
        correnum += np.max(inclunub)
    Purity = correnum / len(predY)
    res,_ = bestMap(Y, predY)
    res=np.array(res)
    Y=np.array(Y)
    ACC=0
    """
    for i in range(len(res)):
        if Y[i]==res[i]:
            ACC=ACC+1
    ACC=ACC/len(Y)
    """
    ACC= np.sum(Y == res) / len(Y)
    
    MIhat = MutualInfo(Y, res)
    result = [ACC, MIhat, Purity]
    return result


# In[18]:


Dataname = 'bbcsport4vbigRnSp'
percentDel = 0.5
para_r = 2
para_k = 5
para_k2 = 5
lambda1 = 0.0001
lambda2 = 0.001
f = 3

Datafold = f"{Dataname}_percentDel_{percentDel}.mat"
data=scio.loadmat(Dataname+'.mat')
X= np.array(data['X'])
truth= np.array(data['truth'])
folds=scio.loadmat(Datafold)
folds= np.array(folds['folds'])
truthF = truth.copy()
ind_folds = folds[0][f-1]
ind_folds=np.array(ind_folds)
numClust = len(np.unique(truthF))
num_view = len(X[0])
NumSamp = len(truthF)

if X[0][0].shape[1] != NumSamp:
    for iv in range(num_view):
        X[iv] = X[iv].T

linshi_WW = 0
Y=[]
G=[]
Wiv=[]
for iv in range(num_view):
    X1 = X[0][iv]
    X1 = NormalizeFea(X1, 0)
    ind_0 = np.where(ind_folds[:, iv] == 0)[0]
    X1[:, ind_0] = 0
    X[0][iv] = X1
    X1 = np.delete(X1, ind_0, axis=1)
    Y.append(X1)

    W1 = np.eye(ind_folds.shape[0])
    W1 = np.delete(W1, ind_0, axis=0)
    G.append(W1)
    linshi_W = np.ones((NumSamp, NumSamp))
    linshi_W[ind_0, :] = 0
    linshi_W[:, ind_0] = 0
    Wiv.append(linshi_W)
Y=np.array(Y)

Sk_ini=[]
Sb_ini=[]
for iv in range(num_view):
    options = {}
    options['NeighborMode'] = 'KNN'
    options['k'] = para_k
    options['WeightMode'] = 'HeatKernel'  # HeatKernel
    Y_iv = Y[iv].T  
    Z1 = constructW(Y_iv, options).toarray()
    Sk_ini.append(G[iv].T.dot(Z1).dot(G[iv]))

    options['WeightMode'] = 'Binary'  # Binary
    options['k'] = para_k2
    Z1 = constructW(Y_iv, options).toarray()
    Z1 = Z1 + np.eye(Z1.shape[0])
    Z1 = Z1.dot(Z1)
    Z1 = Z1 / np.max(Z1)
    Sb_ini.append(G[iv].T.dot(Z1).dot(G[iv]))   # H^(v) in paper
F_ini = solveF(Sk_ini, numClust)
max_iter = 100

F, Z, obj = HCLS_CGL(Sk_ini, Sb_ini, Wiv, F_ini, numClust, lambda1, lambda2, para_r, max_iter)
new_F = F.copy()
temp=np.sum(new_F*new_F, axis=1)
temp=Tensor(temp,mstype.float64)
temp=np2.sqrt(temp)
temp=temp.asnumpy()
norm_mat = np.tile(temp.reshape(-1, 1), (1, new_F.shape[1]))
norm_mat[norm_mat == 0] = 1
new_F = new_F / norm_mat

pre_labels = KMeans(n_clusters=numClust, n_init=20).fit_predict(np.real(new_F))
pre_labels=np.array(pre_labels)
pre_labels=pre_labels+1
pre_labels = np.reshape(pre_labels,(len(pre_labels),1))
result_cluster = ClusteringMeasure(truthF, pre_labels)
for i in range(len(result_cluster)):
    result_cluster[i]=result_cluster[i]*100
print(result_cluster)




