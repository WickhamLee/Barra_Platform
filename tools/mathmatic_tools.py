import numpy as np
import copy
import time
import warnings
warnings.filterwarnings("ignore")

def numpy_rolling(a, window):
    '''
    Parameters
    ----------
    a : numpy.matrix
        DESCRIPTION.
    window : int
        DESCRIPTION.

    Returns
    -------

        对numpy矩阵实现同pandas.rolling一样的功能.

    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def cov_between_mtx(mt1, mt2):
    '''
    

    Parameters
    ----------
    mt1 : numpy.matrix
        DESCRIPTION.
    mt2 : numpy.matrix
        DESCRIPTION.

    Returns
    -------
    TYPE
        计算两个矩阵逐列的协方差.

    '''

    cov_mtx = np.cov(mt1, mt2, rowvar=False)
    return np.diagonal(cov_mtx, offset=-mt1.shape[1])


def array_roll_lookback(array, window = None):
    '''
    

    Parameters
    ----------
    array : int
        DESCRIPTION.
    window : itn, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        对输入的array取每若干个位置上的数，逆序求累加，返回累加得到的数列的最大最小值之差.

    '''
    if window == None: 
        adj_array = np.nancumsum(np.flipud([np.log(array[i*21+19]/array[i*21]) for i in range(12)]))
        return np.nanmax(adj_array) - np.nanmin(adj_array)
    else:
        print('暂不支持输入252以外的时间窗口')
        
def full_array_roll_lookback(array, window):        
    return list(map(lambda x: array_roll_lookback(array[x-window:x]), range(window, len(array))))
    
    
    
    
if __name__ == '__main__':     

    x=np.random.random((100,200))
    y=np.random.random(1000)
    # x=np.array(x)
    # y=np.array(y)
    w=list(np.linspace(0.1,0.5,100))
    w=np.array(w)



    import pandas as pd
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y)
    # r = df_x.rolling(30).apply(f(15))
    # d = df_x.rolling(30).apply(e(15,df_y.rolling(30)))
    
    # tt = time.time()
    # order = range(x.shape[1])
    # roll_x = list(map(lambda i: numpy_rolling(x[:,i], 10).var(axis = 1), order))
    
    
    # print('Time used: {} sec'.format(time.time()-tt))
    # y=np.linspace(1,252,252)
    # ww=full_array_roll_lookback(y, 252)