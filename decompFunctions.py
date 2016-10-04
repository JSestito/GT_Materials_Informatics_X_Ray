# functions
from sklearn import metrics
from sklearn import decomposition 
from sklearn.cluster import KMeans, DBSCAN,MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

def preping(dataList, logscaling = False, feature_scaling = False, **kwargs):
    data = np.array([data for data in dataList])
    qsize = sum([l.shape[0] for l in dataList])
    xsize, ysize = dataList[0].shape[1], dataList[0].shape[-1]
    data = data.reshape(qsize, ysize, xsize)
#     logData = np.log10(data)
    #logData = data
    if logscaling:
        Data = np.log(data)
    else:
        Data = data
    if feature_scaling:
        normData = np.array([scaling(imp, **kwargs) for imp in Data])
    else:
        normData = Data
    normData = normData.reshape(qsize,-1)
    return normData
    
def calSilhouette(data, clustRange = 20):
    met = []
    for clus in np.arange(2,clustRange):
        k_means = KMeans(init='k-means++', n_clusters=clus, n_init=10)
        k_means.fit(data)
        labels = k_means.labels_
        score =  metrics.silhouette_score(data, labels, metric='euclidean')
        met.append(score)
    fig = plt.figure()
    plt.plot(np.arange(2,clustRange), met, marker = 'o')
    plt.xlabel('# clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.show
    return met

def getKmeansClusters(data, nClusters,**kwargs):
    #nClusters = 8
    k_means = KMeans(init='k-means++', n_clusters=nClusters, n_init=10)
    k_means.fit(data)
    clustCents = k_means.cluster_centers_
    size = int(np.sqrt(clustCents.shape[-1]))
    pltData = clustCents.reshape(-1,size,size)
    fig, axes = plt.subplots(nClusters/4,4, figsize = (12,12))
    fig.subplots_adjust(wspace=0.1, hspace=0.1,bottom = 0., top=0.5)
    for ax, imp, nClus in zip(axes.flat, pltData, np.arange(nClusters)):
        ax.imshow(imp, **kwargs)
        label = 'Cluster = %d' %(nClus+1)
        ax.set_title(label)
        ax.axis('off')
    return pltData, k_means
    
def doPCA(data, nComponents, xvals = [], xlabel = '', **kwargs):
    '''
    Does probabilistic PCA on data.
    
    Parameters:
    -----------
    data: output of preping function or ndarray with shape = (n_samples, n_features)
    nComponents: # of principal axes to use.
    xvals: Values to use for plotting the eigenvectors. Default is empty list.
    xlabel: label for the x-axis. Default is ''.
    **kwargs: keyword arguments passed to matplotlib ax.imshow() function.
    
    Returns:
    --------
    projected data, eigenvectors, list of explained variance ratio, PCA estimator object.
    
    '''
    pca = decomposition.PCA(whiten = False, n_components = nComponents)
    eigenVals = pca.fit_transform(data)
    comps = pca.components_
    size = int(np.sqrt(comps.shape[-1]))
    pltData = comps.reshape(-1,size,size)
    varRatio = pca.explained_variance_ratio
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12))
    fig.subplots_adjust(wspace=0.1, hspace=0.1,bottom = 0., top=0.5)
#     fig.subplots_adjust(left=-0.01, bottom=-0.01, right=0.01, top=0.01,
#                     wspace=0.01, hspace=0.01)
    for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
        ax.imshow(imp, **kwargs)
        label = 'Component = %d' %(nComp+1)
        ax.axis('off')
        ax.set_title(label)
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12),sharex = True, sharey = False)
    fig.subplots_adjust(hspace=0.3)
    for ax, nComp, ratio in zip(axes.flat, np.arange(nComponents), varRatio):
        if len(xvals) is not 0:
            ax.plot(xvals, eigenVals[:,nComp],marker='o',markerfacecolor='r')
        else:
            ax.plot(eigenVals[:,nComp],marker='o',markerfacecolor='r')
        label = 'Component = %d, Exp. Var. Ratio = %2.3f  ' %(nComp+1, ratio)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        
    return np.array(pltData), np.array(eigenVals), np.array(varRatio), pca
    
def doNMF(data, nComponents,  xvals = [], xlabel='', **kwargs):
    '''
    Does NMF on data.
    
    Parameters:
    -----------
    data: output of preping function or ndarray with shape = (n_samples, n_features)
    nComponents: # of principal axes to use.
    xvals: Values to use for plotting the eigenvectors. Default is empty list.
    xlabel: label for the x-axis. Default is ''.
    **kwargs: keyword arguments passed to matplotlib ax.imshow() function.
    
    Returns:
    --------
    projected data, encodings, NMF estimator object.
    
    '''
    nmf = decomposition.NMF(n_components = nComponents, max_iter=int(1e6),init='random')
    dat = nmf.fit_transform(data)
#     return dat, nmf
    comps = nmf.components_
    size = int(np.sqrt(comps.shape[-1]))
    pltData = comps.reshape(-1,size,size)
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12))
    fig.subplots_adjust(wspace=0.1, hspace=0.01,bottom = 0., top=0.5)
#     fig.subplots_adjust(left=-0.01, bottom=-0.01, right=0.01, top=0.01,
#                     wspace=0.01, hspace=0.01)
    for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
        ax.imshow(imp, **kwargs)
        label = 'Component = %d' %(nComp+1)
        ax.axis('off')
        ax.set_title(label)
    plt.show()
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, sharex = True, sharey = False, figsize = (12,12))
    fig.subplots_adjust(hspace=0.3)
#     imageTile(pltData)
    encod = dat.T
    for ax, enc,nComp in zip(axes.flat, encod,np.arange(nComponents)):
        if len(xvals) is not 0:
            ax.plot(xvals, enc,marker='o',markerfacecolor='r')
        else:
            ax.plot(enc,marker='o',markerfacecolor='r')
        label = 'Component = %d' %(nComp+1)
        ax.set_title(label)  
        ax.set_xlabel(xlabel)
    plt.show()
    return np.array(pltData), np.array(encod), nmf

def doICA(data, nComponents, xvals = [], xlabel = '', **kwargs):
    ica = decomposition.FastICA(whiten = True, n_components=nComponents, fun='logcosh')
    eigenVals = ica.fit_transform(data)
    comps = ica.components_
    size = int(np.sqrt(comps.shape[-1]))
    pltData = comps.reshape(-1,size,size)
#     varRatio = ica.explained_variance_imgs,eigenvals,_,pca
#     vis_square(pltData, padsize=1, padval=0, cmap='jet', vmax = 1e-5)
    fig, axes = plt.subplots(nComponents/4,4, figsize = (12,12))
    fig.subplots_adjust(wspace=0.1, hspace=0.1,bottom = 0., top=0.5)
#     fig.subplots_adjust(left=-0.01, bottom=-0.01, right=0.01, top=0.01,
#                     wspace=0.01, hspace=0.01)
    for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
        ax.imshow(imp, **kwargs)
        label = 'Component = %d' %(nComp+1)
        ax.axis('off')
        ax.set_title(label)
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12),sharex = True, sharey = False)
    fig.subplots_adjust(hspace=0.3)
    for ax, nComp in zip(axes.flat, np.arange(nComponents)):
        if len(xvals) is not 0:
            ax.plot(xvals, eigenVals[:,nComp],marker='o',markerfacecolor='r')
        else:
            ax.plot(eigenVals[:,nComp],marker='o',markerfacecolor='r')
        label = 'Component = %d' %(nComp+1)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        
    return pltData, eigenVals, ica