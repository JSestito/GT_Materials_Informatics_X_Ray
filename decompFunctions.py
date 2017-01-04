# functions
from sklearn import metrics
from sklearn import decomposition 
from sklearn.cluster import KMeans, DBSCAN,MiniBatchKMeans
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def preping(data, logscaling = False, feature_scaling = False, **kwargs):
    qsize = data.shape[0]
    xsize, ysize = data.shape[1],data.shape[-1]
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
    
def doPCA(data, nComponents, plot = True, xvals = [], xlabel = '',normalize = False, returnComponents = -1, **kwargs):
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

    if normalize == True:
        #Find the Mean
        meanvect = np.mean(data, axis = 1)
        #Normalize Data
        data = np.subtract(data.transpose(),meanvect).transpose()
        
    if returnComponents < 0:
        returnComponents = nComponents
    
    
    pca = decomposition.PCA(whiten = False, n_components = returnComponents)
    eigenVals = pca.fit_transform(data)
                                          
    comps = pca.components_
    size = int(np.sqrt(comps.shape[-1]))
    
    if plot == True:
        pltData = comps.reshape(-1,size,size)
    else:
        pltData = comps
    
    varRatio = pca.explained_variance_ratio_
    
    if plot == True:
        if normalize == True:
            fig,ax = plt.subplots()
            ax.set_title('Mean')
            ax.plot(xvals,meanvect,marker = 'o', markerfacecolor='r')
            ax.set_xlabel('$2\\theta$ (deg.)')
            ax.set_ylabel('(Intensity) $ (arb. units)')
    
        fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12))
    #ig.subplots_adjust(wspace=0.1, hspace=0.1,bottom = 0., top=0.5)
    
#     fig.subplots_adjust(left=-0.01, bottom=-0.01, right=0.01, top=0.01,
#                     wspace=0.01, hspace=0.01)
        for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
            im = ax.imshow(imp, **kwargs)
            label = 'Component = %d' %(nComp+1)
            ax.axis('off')
            ax.set_title(label)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
   
        fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12),sharex = True, sharey = False)
        fig.subplots_adjust(hspace=0.3)
        for ax, nComp, ratio in zip(axes.flat, np.arange(nComponents), varRatio):
            if len(xvals) is not 0:
                ax.plot(xvals[:], eigenVals[:,nComp],marker='o',markerfacecolor='r')
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
    #fig.subplots_adjust(wspace=0.1, hspace=0.01,bottom = 0., top=0.5)
#     fig.subplots_adjust(left=-0.01, bottom=-0.01, right=0.01, top=0.01,
#                     wspace=0.01, hspace=0.01)
    for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
        im = ax.imshow(imp, **kwargs)
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



def doPCA_Mask(data, nComponents, mask3D, xvals = [], xlabel = '',normalize = True, returnComponents = -1, **kwargs):
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

    mask = preping(mask3D, logscaling = False, feature_scaling=False, feature_range=(0.1,1))
    
    
    data_mask = np.zeros((mask.shape[0],int(mask.shape[1] - np.sum(mask[0,:]))))
    
    j = 0
    for i in range(0,mask.shape[1]):
        if mask[0,i] == 0:
            data_mask[:,j] = data[:,i]
            j = j+1
    
    
    #Find the Mean
    meanvect = np.mean(data_mask, axis = 1)
    #Normalize Data
    data_mask = np.subtract(data_mask.transpose(),meanvect).transpose()
        
    if returnComponents < 0:
        returnComponents = nComponents
    
    
    pca = decomposition.PCA(whiten = False, n_components = returnComponents)
    eigenVals = pca.fit_transform(data_mask)
                                          
    comps = pca.components_
    varRatio = pca.explained_variance_ratio_
    
    comps_mask = np.zeros((comps.shape[0],mask.shape[1]))
    j = 0
    for i in range(0,mask.shape[1]):
        if mask[0,i] == 0:
            comps_mask[:,i] = comps[:,j]
            j = j + 1
   
   
    
    fig,ax = plt.subplots()
    ax.set_title('Mean')
    ax.plot(xvals,meanvect,marker = 'o', markerfacecolor='r')
    ax.set_xlabel('$2\\theta$ (deg.)')
    ax.set_ylabel('(Intensity) $ (arb. units)')

    size = int(np.sqrt(comps_mask.shape[-1]))
    pltData = comps_mask.reshape(-1,size,size)
    
    
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12))
    for ax, imp, nComp in zip(axes.flat, pltData, np.arange(nComponents)):
        im = ax.imshow(ma.masked_array(imp,mask3D[0,:,:], **kwargs))
        label = 'Component = %d' %(nComp+1)
        ax.axis('off')
        ax.set_title(label)
        
      
    fig, axes = plt.subplots(nComponents/2,2+nComponents%2, figsize = (12,12),sharex = True, sharey = False)
    fig.subplots_adjust(hspace=0.3)
    for ax, nComp, ratio in zip(axes.flat, np.arange(nComponents), varRatio):
        if len(xvals) is not 0:
            ax.plot(xvals[:], eigenVals[:,nComp],marker='o',markerfacecolor='r')
        else:
            ax.plot(eigenVals[:,nComp],marker='o',markerfacecolor='r')
            
        label = 'Component = %d, Exp. Var. Ratio = %2.3f  ' %(nComp+1, ratio)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        
    return np.array(pltData), np.array(eigenVals), np.array(varRatio), pca