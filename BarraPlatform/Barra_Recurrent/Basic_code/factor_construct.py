import numpy as np
import pandas as pd
from tqdm import tqdm
# import statsmodels.api as sm
# import time
from multiprocessing import Pool

from tools.mathmatic_tools import Weighted_Linear_Regression, Rolling_Weighted_Linear_Regression, Rolling_Weighted_Linear_Regression_Merge_Config
from tools.signals_general import reldiff, std_roll
from Basic_code.data_load import config_load, basic_data_load


config = config_load()
basic_data = basic_data_load()
industry_name_dic = {'1': '农林牧渔',
                     '2': '采掘',
                     '3': '化工',
                     '4': '钢铁',
                     '5': '有色金属',
                     '6': '电子',
                     '7': '家用电器',
                     '8': '食品饮料',
                     '9': '纺织服装',
                     '10': '轻工制造',
                     '11': '医药生物',
                     '12': '公用事业',
                     '13': '交通运输',
                     '14': '房地产',
                     '15': '商业贸易',
                     '16': '休闲服务',
                     '17': '综合',
                     '18': '建筑材料',
                     '19': '建筑装饰',
                     '20': '电气设备',
                     '21': '国防军工',
                     '22': '计算机',
                     '23': '传媒',
                     '24': '通信',
                     '25': '银行',
                     '26': '非银金融',
                     '27': '汽车',
                     '28': '机械设备'}


#构建行业因子
class Industry_factor:
    
    def __init__(self):      
        self.batch_run()
    # +-------------------------------------------------------+
    # |                     ind_fac                           |
    # +-------------------------------------------------------+
    #构建行业因子暴露，由0和1组成
    def ind_fac(self, ind):
        # global config, basic_data
        df_ind_fac = pd.DataFrame(np.zeros(basic_data['close'].shape),
                                  index = basic_data['close'].index,
                                  columns = basic_data['close'].columns)
        df_ind_fac[basic_data[config['Industry_Class'][0]].values == ind] = 1
        return df_ind_fac

    # +-------------------------------------------------------+
    # |                    batch_run                          |
    # +-------------------------------------------------------+
    #构建行业因子暴露，由0和1组成
    def batch_run(self):
        # global industry_name_dic
        for i in tqdm(industry_name_dic, desc = '开始生成行业因子暴露'):
            exec('self.industry_' +i+ '=self.ind_fac(int(' +i+ '))')


            
# 构建风格因子
class Style_factor:

    def __init__(self):
        self.ret_hs_300 = reldiff(basic_data['000300'])['close']
        self.stock_return = reldiff(basic_data['close'])
    # +-------------------------------------------------------+
    # |                       SIZE                            |
    # +-------------------------------------------------------+
    #市值因子，包含LNCAP和MIDCAP
    def SIZE(self):
        # LNCAP因子
        print('开始构建LNCAP因子')
        LNCAP = np.log(basic_data['market_cap'])
        
        # MIDCAP因子
        LNCAP_cubic = LNCAP**3
        MIDCAP = np.zeros((1, LNCAP_cubic.shape[1]))
        
        for i in tqdm(range(LNCAP_cubic.shape[0]), desc = '开始构建MIDCAP因子'):
            x = LNCAP.values[i,:]
            y = LNCAP_cubic.values[i,:]
            reg = np.polyfit(x[~np.isnan(x)], y[~np.isnan(y)], 1)
            res = y - (x*reg[0] + reg[1])
            MIDCAP = np.vstack((MIDCAP, res))
            
        return LNCAP, pd.DataFrame(MIDCAP[1:,:], index = LNCAP.index, columns = LNCAP.columns)

    # +-------------------------------------------------------+
    # |                    VOLATILITY                         |
    # +-------------------------------------------------------+
    #波动率因子，包含BETA,HIST_SIGMA,DAILY_STD,CUMULATIVE_RANGE因子
    def VOLATILITY(self):
        # BETA因子       
        weights_half_life = 0.5**(np.linspace(251,0,252)/63)
        xx=self.stock_return.values
        xx[xx == 0] = np.nan
        yy=self.ret_hs_300.values

        result_beta, result_alpha = [], []
        
        with Pool(processes = 8) as p:
            max_ = self.stock_return.shape[1]
            with tqdm(total = max_, desc = '开始构建BETA和HIST_SIGMA因子') as pbar:
                for v in p.imap_unordered(
                        Rolling_Weighted_Linear_Regression_Merge_Config, 
                                                        [[xx[:, j], yy, 252, weights_half_life] for j in range(self.stock_return.shape[1])]
                                                        ):
                    result_beta.append(v[0])
                    result_alpha.append(v[1])
                    pbar.update()

        BETA = pd.DataFrame(result_beta, index = self.stock_return.columns, columns = self.stock_return.index).T             #xx[:,2937], yy
        alpha_df = pd.DataFrame(result_alpha, index = self.stock_return.columns, columns = self.stock_return.index).T   
       
        # HIST_SIGMA因子 
        HIST_SIGMA = std_roll(BETA*xx + alpha_df - yy.reshape(len(yy), 1), 20)
       
        return BETA, HIST_SIGMA