import numpy as np
import copy
import time
import warnings
warnings.filterwarnings("ignore")
# +-------------------------------------------------------+
# |           Weighted_Linear_Regression                  |
# +-------------------------------------------------------+
#加权线性回归
def Weighted_Linear_Regression(array_x, array_y, array_weight):
    if np.isnan(array_x[15:-15]).all():
        return [[np.nan], [np.nan]]
    else:
        reference = copy.deepcopy(array_x)
        array_x = array_x[~np.isnan(reference)]
        array_y = array_y[~np.isnan(reference)]
        array_weight = array_weight[~np.isnan(reference)]
            
        ss = [1]*len(array_x)
        ss=np.array(ss).reshape(len(array_x),1)
        x = array_x.reshape(len(array_x), 1)
        x = np.hstack((ss,x))
        y = array_y.reshape(len(array_y), 1)
    
        weight = np.mat(np.eye((len(x))))
        weight[np.diag_indices_from(weight)] = array_weight
              
        end = (x.T*(weight*x)).I*(x.T*(weight*y))
        end = end.getA().tolist()#矩阵转成列表      
        return end
            
            
# +-------------------------------------------------------+
# |       Rolling_Weighted_Linear_Regression              |
# +-------------------------------------------------------+
#滚动加权线性回归
def Rolling_Weighted_Linear_Regression(array_x, array_y, window, array_weight):
    reg_beta, reg_alpha = [], []
    list_x, list_y = list(array_x), list(array_y)
    x, y = list_x[:window], list_y[:window]
    
    for i in range(window, len(array_x)):

        result = Weighted_Linear_Regression(np.array(x[i-window:i]), np.array(y[i-window:i]), array_weight)
        reg_beta.append(result[0][0])
        reg_alpha.append(result[1][0])

        x += [list_x[i]]
        y += [list_y[i]]

    result = Weighted_Linear_Regression(np.array(x[-window:]), np.array(y[-window:]), array_weight)
    reg_beta.append(result[0][0])
    reg_alpha.append(result[1][0])
    return [np.nan]*(window-1) + reg_beta, [np.nan]*(window-1) + reg_alpha

# +-------------------------------------------------------+
# |     Rolling_Weighted_Linear_Regression_Merge_Config   |
# +-------------------------------------------------------+
#滚动加权线性回归的参数整合版
def Rolling_Weighted_Linear_Regression_Merge_Config(list):
    return Rolling_Weighted_Linear_Regression(list[0], list[1], list[2], list[3])


 
if __name__ == '__main__':
    # Weighted_Linear_Regression函数的单元测试，结果与sm.WLS的计算结果是一致的
    import statsmodels.api as sm            

    x=np.random.random(100)
    # x, y =[np.nan]*12, []*12
    y=np.random.random(100)
    # x=np.array(x)
    # y=np.array(y)
    w=list(np.linspace(0.1,0.5,100))
    w=np.array(w)

    # st1 = time.time()
    # b=sm.WLS(y, sm.add_constant(x), weights=w).fit()
    # ed1 = time.time()
    # print('计算结果：', b.params, '耗时：', (ed1-st1), 's')

    
    # st2 = time.time()
    # a=Weighted_Linear_Regression(x, y, w)
    # ed2 = time.time()
    # print('计算结果：', a, '耗时：', (ed2-st2), 's')

    
    tt = time.time()
    c = Rolling_Weighted_Linear_Regression(x, y, 5, w)
    print(c)
    print('Time used: {} sec'.format(time.time()-tt))
    import pandas as pd
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y)
    d = df_x.rolling(5).corr(df_y, pairwise = True)
    