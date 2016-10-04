import numpy as np
from matplotlib import pyplot as plt

def imageTile(data, padsize=1, padval=0, figsize=(12,12),**kwargs):
    '''
    Function to tile n-images into a single image   
    Input:  
    data: np.ndarray. Must be of dimension 3.  
    padsize: size in pixels of borders between different images. Default is 1.  
    padval: value by which to pad. Default is 0. 
    figsize: size of the figure passed to matplotlib.pyplot.figure. Default is (12,12).  
    **kwargs: extra arguments to be passed to pyplot.imshow() function.
    '''
   
    # force the number of images to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile all the images into a single image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(data,**kwargs)
    ax.axis('off')

def scaling(data, feature_range=(0,1)):
    '''
    Scaling data between feature_range.
    To retrieve original data: data = scaled*scaleFactor
    Input:
    data (np.array), feature_range(tuple)
    
    Output:
    
    scaled data(np.array), scalingfactor (np.array)
    '''
    data = np.log(data)
    data_std = (data - data.min())/(data.max()- data.min())
    scaled = data_std * (feature_range[-1] - feature_range[0]) + feature_range[0]
    scaleFactor = data/scaled
    return scaled


def atr(data,name,image)
    return dset.attrs[name][image]