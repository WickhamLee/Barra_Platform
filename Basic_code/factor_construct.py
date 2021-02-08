import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import time
from multiprocessing import Pool

# from tools.mathmatic_tools_trash import Rolling_Weighted_Linear_Regression_Merge_Config
from tools.mathmatic_tools import numpy_rolling, cov_between_mtx, full_array_roll_lookback
from tools.signals_general import reldiff, delta, sum_roll, ma, std_roll
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
        self.stock_turnover_ratio = reldiff(basic_data['turnover_ratio'])
    # +-------------------------------------------------------+
    # |                       SIZE                            |
    # +-------------------------------------------------------+
    #市值因子，包含LNCAP和MIDCAP
    def SIZE(self):
        # LNCAP因子
        tt1 = time.time()    
        LNCAP = np.log(basic_data['market_cap'])
        print('LNCAP因子构建完成，耗时: {} sec'.format(time.time()-tt1))
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
             
        weights_half_life = 0.5**(np.linspace(251,0,252)/63)
        weights_half_life_normalize = weights_half_life/weights_half_life.sum()
        xx=self.stock_return.values
        # xx[xx == 0] = np.nan
        yy=self.ret_hs_300.values

        # result_beta, result_alpha = [], []
        
        # with Pool(processes = 8) as p:
        #     max_ = self.stock_return.shape[1]
        #     with tqdm(total = max_, desc = '开始构建BETA和HIST_SIGMA因子') as pbar:
        #         for v in p.imap_unordered(
        #                 Rolling_Weighted_Linear_Regression_Merge_Config, 
        #                                                 [[xx[:, j], yy, 252, weights_half_life] for j in range(self.stock_return.shape[1])]
        #                                                 ):
        #             result_beta.append(v[0])
        #             result_alpha.append(v[1])
        #             pbar.update()

        # BETA = pd.DataFrame(result_beta, index = self.stock_return.columns, columns = self.stock_return.index).T             #xx[:,2937], yy
        # alpha_df = pd.DataFrame(result_alpha, index = self.stock_return.columns, columns = self.stock_return.index).T   
        
        '''
        BETA因子  
        '''
        tt1 = time.time()          
        order = range(xx.shape[1])
        roll_xx = list(map(lambda i: numpy_rolling(xx[:,i], 252)*10000, order))
        weighted_roll_xx_mean = list(map(lambda i: list((roll_xx[i]*weights_half_life_normalize).sum(axis=0)), order))
        weighted_roll_xx_var = list(map(lambda i: (((roll_xx[i]-weighted_roll_xx_mean[i]).T**2)*weights_half_life_normalize.reshape(252,1)).sum(axis=0), order))

        roll_yy = (numpy_rolling(yy, 252)*weights_half_life).T
        weighted_roll_xy_cov = list(map(lambda i: cov_between_mtx(roll_xx[i].T, roll_yy)/10000, order))
        weighted_roll_xx_var = np.vstack((np.zeros((251, xx.shape[1])), np.mat(weighted_roll_xx_var).T))
        weighted_roll_xy_cov = np.vstack((np.zeros((251, xx.shape[1])), np.mat(weighted_roll_xy_cov).T))
        BETA = pd.DataFrame(weighted_roll_xy_cov/weighted_roll_xx_var, index = self.stock_return.index, columns = self.stock_return.columns)
        print('BETA因子构建完成，耗时: {} sec'.format(time.time()-tt1))
        '''
        HIST_SIGMA因子  
        ''' 
        tt2 = time.time() 
        alpha_volitatility = list(map(lambda i: np.nanstd(yy[i-251:i].reshape(251,1)-BETA.values[i]*xx[i-251: i,:], axis=0), range(251, xx.shape[0])))
        HIST_SIGMA_mtx = np.vstack((np.zeros((251, xx.shape[1])), np.mat(alpha_volitatility)))
        self.HIS_ALPHA = list(map(lambda i: np.nanmean(yy[i-251:i].reshape(251,1)-BETA.values[i]*xx[i-251: i,:], axis=0), range(251, xx.shape[0])))
        self.time_HIS_ALPHA = time.time()-tt2
        HIST_SIGMA = pd.DataFrame(HIST_SIGMA_mtx, index = self.stock_return.index, columns = self.stock_return.columns)
        print('HIST_SIGMA因子构建完成，耗时: {} sec'.format(time.time()-tt2))

        '''
        DAILY_STD因子  
        ''' 
        tt3 = time.time() 
        DAILY_STD = pd.DataFrame(np.sqrt(weighted_roll_xx_var), index = self.stock_return.index, columns = self.stock_return.columns)
        print('DAILY_STD因子构建完成，耗时: {} sec'.format(time.time()-tt3))
        
        '''
        CUMULATIVE_RANGE因子  
        '''  
        tt4 = time.time() 
        roll_252lb_xx = list(map(lambda x: full_array_roll_lookback(basic_data['close'].values[:,x], 251), order))
        CUMULATIVE_RANGE = pd.DataFrame(np.vstack((np.zeros((251, xx.shape[1])), np.mat(roll_252lb_xx).T)),
                                     index = self.stock_return.index, columns = self.stock_return.columns)   
        
        print('CUMULATIVE_RANGE因子构建完成，耗时: {} sec'.format(time.time()-tt4))
        return BETA, HIST_SIGMA, DAILY_STD, CUMULATIVE_RANGE
    
    # +-------------------------------------------------------+
    # |                     LIQUIDITY                         |
    # +-------------------------------------------------------+
    #流动性因子，包含STOM(月换手率),STOQ(季换手率),STOA(年换手率),SUMSTOA(年化交易量比率)因子
    def LIQUIDITY(self):
        '''
        STOM因子  
        '''  
        tt1 = time.time()  
        STOM = pd.DataFrame(np.log(sum_roll(self.stock_turnover_ratio, 21)), 
                            index = self.stock_return.index, columns = self.stock_return.columns)   
        print('STOM因子构建完成，耗时: {} sec'.format(time.time()-tt1))
        '''
        STOQ因子  
        '''  
        tt2 = time.time() 
        STOQ = pd.DataFrame(np.log(sum_roll(np.exp(STOM), 21*3)/3), 
                            index = self.stock_return.index, columns = self.stock_return.columns)  
        print('STOQ因子构建完成，耗时: {} sec'.format(time.time()-tt2))
        '''
        STOA因子  
        '''  
        tt3 = time.time() 
        STOA = pd.DataFrame(np.log(sum_roll(np.exp(STOM), 21*12)/12), 
                            index = self.stock_return.index, columns = self.stock_return.columns)  
        print('STOA因子因子构建完成，耗时: {} sec'.format(time.time()-tt3))
        '''
        SUMSTOM因子  
        '''  
        tt4 = time.time() 
        weights_half_life = 0.5**(np.linspace(251,0,252)/63)
        weights_half_life_normalize = weights_half_life/weights_half_life.sum() 
        order = range(self.stock_turnover_ratio.shape[1])
        turnover_ = self.stock_turnover_ratio.values
        roll_stock_turnover_ratio = list(map(lambda i: numpy_rolling(turnover_[:,i], 252), order))
        SUMSTOM_list = list(map(lambda i: (roll_stock_turnover_ratio[i]*weights_half_life_normalize).sum(axis=1), order))
        SUMSTOM = pd.DataFrame(np.vstack((np.zeros((251, self.stock_turnover_ratio.shape[1])), np.mat(SUMSTOM_list).T)), index = self.stock_return.index, columns = self.stock_return.columns)
        print('SUMSTOM因子构建完成，耗时: {} sec'.format(time.time()-tt4))
        
        return STOM, STOQ, STOA, SUMSTOM
        
    # +-------------------------------------------------------+
    # |                     MOMENTUM                          |
    # +-------------------------------------------------------+
    #动量因子，包含STREV(短期反转),SEASON(季节因子),INDMOM(行业动量),RSTR(相对于市场的强度), HIS_ALPHA(历史Alpha)因子
    def MOMENTUM(self):   
        '''
        STREV因子  
        '''  
        tt1 = time.time()
        weights_half_life = 0.5**(np.linspace(20,0,21)/5)
        weights_half_life_normalize = weights_half_life/weights_half_life.sum()
        xx=self.stock_return.values
        order = range(xx.shape[1])
        roll_xx = list(map(lambda i: numpy_rolling(xx[:,i], 21), order))
        weighted_roll_xx_mean = list(map(lambda i: (roll_xx[i]*weights_half_life_normalize).sum(axis=1), order))
        weighted_roll_xx_mean = np.vstack((np.zeros((20, xx.shape[1])), np.mat(weighted_roll_xx_mean).T))
        STREV = pd.DataFrame(weighted_roll_xx_mean, index = self.stock_return.index, columns = self.stock_return.columns)
        print('STREV因子构建完成，耗时: {} sec'.format(time.time()-tt1))

        '''
        SEASON因子  
        ''' 
        tt2 = time.time() 
        stock_daily_close = basic_data['close'].copy()
        stock_daily_close['date'] = stock_daily_close.index
        stock_daily_close["year"] = pd.DatetimeIndex(stock_daily_close["date"]).year
        stock_daily_close["month"] = pd.DatetimeIndex(stock_daily_close["date"]).month
        gb = stock_daily_close.groupby(['year', 'month'])
        adj_gb = reldiff(gb.apply(lambda _df: _df.iloc[-1,:-3])) 
        adj_gb = pd.DataFrame(ma(adj_gb, 60), index = adj_gb.index, columns = adj_gb.columns)
        SEASON = pd.DataFrame(index = self.stock_return.index, columns = self.stock_return.columns)
        for i in range(2015, stock_daily_close["year"][-1]+1):
            for j in range(1,13):
                day = 1
                if (dt.datetime(i, j, day) - dt.datetime.strptime(SEASON.index[-1], '%Y-%m-%d')).days<0:
                    while (dt.date(i, j, day).strftime('%Y-%m-%d') not in list(SEASON.index)):
                        day+=1
                    SEASON.loc[dt.date(i, j, day).strftime('%Y-%m-%d')] = adj_gb.loc[i,j]
        SEASON = SEASON.ffill()
        print('SEASON因子构建完成，耗时: {} sec'.format(time.time()-tt2))
        
        '''
        INDMOM因子  
        '''   
        tt3 = time.time() 
        weights_half_life = 0.5**(np.linspace(125,0,126)/21)
        weights_half_life_normalize = weights_half_life/weights_half_life.sum() 
        roll_xx = list(map(lambda i: numpy_rolling(xx[:,i], 126), order))
        # 个股相对强度
        RS_s_list = list(map(lambda i: (roll_xx[i]*weights_half_life_normalize).sum(axis=1), order))
        RS_s = np.vstack((np.zeros((125, xx.shape[1])), np.mat(RS_s_list).T))
        market_cap = basic_data['market_cap']
        ind_class = basic_data['sw_l1']
        c_s = np.sqrt(market_cap)
        # 行业I_t的相对强度
        RS_i_list = list(map(lambda x: np.nansum(c_s[ind_class==x]*self.stock_return[ind_class==x], axis=1), range(1,29)))
        RS_i_list += [[np.nan]*len(RS_i_list[0])]
        tool_mtx1 = np.multiply(c_s.values, pd.DataFrame(RS_s).values)
        x=list(map(lambda i: tool_mtx1[:,i], range(tool_mtx1.shape[1])))
        y = list(map(lambda i: RS_i_list[int(np.unique(np.append(basic_data['sw_l1'].values[:,i], 29))[0])-1], range(tool_mtx1.shape[1])))
        INDMOM = pd.DataFrame((np.mat(x)-np.mat(y)).T, index = self.stock_return.index, columns = self.stock_return.columns)
        print('INDMOM因子构建完成，耗时: {} sec'.format(time.time()-tt3))
        '''
        RSTR因子  
        '''  
        tt4 = time.time() 
        weights_half_life = 0.5**(np.linspace(251,0,252)/126)
        weights_half_life_normalize = weights_half_life/weights_half_life.sum()
        roll_xx = list(map(lambda i: numpy_rolling(xx[10:,i], 252), order))
        RSTR = list(map(lambda i: (np.log(1+roll_xx[i])*weights_half_life_normalize).sum(axis=1), order))
        RSTR = np.vstack((np.zeros((9+252, xx.shape[1])), np.mat(RSTR).T))
        RSTR = pd.DataFrame(RSTR, index = self.stock_return.index, columns = self.stock_return.columns)
        print('RSTR因子构建完成，耗时: {} sec'.format(time.time()-tt4))
        '''
        HIS_ALPHA因子  
        '''  
        HIS_ALPHA = pd.DataFrame(np.vstack((np.zeros((251, xx.shape[1])), np.mat(self.HIS_ALPHA))),
                         index = self.stock_return.index, columns = self.stock_return.columns)        
        print('HIS_ALPHA因子构建完成，耗时: {} sec'.format(self.time_HIS_ALPHA))
        return STREV, SEASON, INDMOM, RSTR, HIS_ALPHA
        
    # +-------------------------------------------------------+
    # |                      QUALITY                          |
    # +-------------------------------------------------------+
    #质量因子，包含MLEV(市场杠杆),BLEV(账面杠杆),DTOA(资产负债比),SALES_VOL(营业收入波动率)因子,EAR_VOL(盈利波动率), 
    # CF_VOL(现金流波动率), FEP_VOL(分析师预测盈市率标准差),ABS(资产负债表应计项目),ACF(现金流量表应计项目),ATO(资产周转率),
    # GP(资产毛利率), GPM(销售毛利率),ROA(总资产收益率),TAG(总资产增长率),IG(股票发行量增长率),CPG(资本支出增长率)
    def QUALITY(self):   
        '''
        MLEV因子  
        '''  
        tt1 = time.time() 
        # MLEV
        print('MLEV因子构建完成，耗时: {} sec'.format(time.time()-tt1))
        '''
        BLEV因子  
        '''  
        tt2 = time.time() 
     
        print('BLEV因子构建完成，耗时: {} sec'.format(time.time()-tt2))
        '''
        DTOA因子  
        '''  
        tt3 = time.time() 
        DTOA = basic_data['total_liability']/basic_data['total_assets']
        print('DTOA因子构建完成，耗时: {} sec'.format(time.time()-tt3))
        '''
        TOR_VOL因子  
        '''  
        tt4 = time.time() 
        TOR_VOL = std_roll(basic_data['total_operating_revenue'], 252*5)/ma(basic_data['total_operating_revenue'], 252*5)
        TOR_VOL = pd.DataFrame(TOR_VOL, index = self.stock_return.index, columns = self.stock_return.columns)

        print('TOR_VOL因子构建完成，耗时: {} sec'.format(time.time()-tt4))
        '''
        NP_VOL因子  
        '''  
        tt5 = time.time() 
        NP_VOL = std_roll(basic_data['net_profit'], 252*5)/ma(basic_data['net_profit'], 252*5)
        NP_VOL = pd.DataFrame(NP_VOL, index = self.stock_return.index, columns = self.stock_return.columns)
        print('NP_VOL因子构建完成，耗时: {} sec'.format(time.time()-tt5))
        '''
        CEI_VOL因子  
        '''  
        tt6 = time.time() 
        CEI_VOL = std_roll(basic_data['cash_equivalent_increase'], 252*5)/ma(basic_data['cash_equivalent_increase'], 252*5)
        CEI_VOL = pd.DataFrame(CEI_VOL, index = self.stock_return.index, columns = self.stock_return.columns)
        print('CEI_VOL因子构建完成，耗时: {} sec'.format(time.time()-tt6))
        '''
        FEP_VOL因子  
        '''          
        tt7 = time.time() 
     
        print('FEP_VOL因子构建完成，耗时: {} sec'.format(time.time()-tt7))
        '''
        ABS因子  
        '''  
        tt8 = time.time() 
        # 聚宽除银行行业有non_interest_bearing_liability的数据，其他行业无此数据，暂用0代替
        NOA = (basic_data['total_assets'] - basic_data['cash_and_equivalents_at_end'])-0
        # 聚宽摊销这就等数据不全，暂用0代替
        ACCR_BS = delta(NOA, 1) - 0
        ACCR_BS[ACCR_BS == 0] = np.nan
        ACCR_BS = ACCR_BS.ffill()
        ABS = -ACCR_BS/basic_data['total_assets']
        print('ABS因子构建完成，耗时: {} sec'.format(time.time()-tt8))
        '''
        ACF因子  
        '''  
        tt9 = time.time() 
        # 聚宽摊销这就等数据不全，暂用0代替
        ACCR_CF = basic_data['net_profit']-(basic_data['net_operate_cash_flow']+basic_data['net_invest_cash_flow'])+0
        ACF = -ACCR_CF/basic_data['total_assets']
        print('ACF因子构建完成，耗时: {} sec'.format(time.time()-tt9))
        '''
        ATO因子  
        '''  
        tt10 = time.time() 
        ATO = basic_data['operating_revenue']/basic_data['total_assets']
        print('ATO因子构建完成，耗时: {} sec'.format(time.time()-tt10))
        '''
        GP因子  
        '''  
        tt11 = time.time() 
        GP = (basic_data['operating_revenue'] - basic_data['operating_cost'])/basic_data['total_assets']
        print('GP因子构建完成，耗时: {} sec'.format(time.time()-tt11))
        '''
        GPM因子  
        '''  
        tt12 = time.time() 
        GPM = (basic_data['operating_revenue'] - basic_data['operating_cost'])/basic_data['operating_revenue']
        print('GPM因子构建完成，耗时: {} sec'.format(time.time()-tt12))
        '''
        ROA因子  
        '''  
        tt13 = time.time() 
        ROA = basic_data['roa']
        print('ROA因子构建完成，耗时: {} sec'.format(time.time()-tt13))
        '''
        TAG因子  
        '''  
        tt14 = time.time() 
        cap = basic_data['total_assets'].values/1000000
        adj_cap_var = std_roll(basic_data['total_assets'].values, 252*5)**2
        adj_cap = list(map(lambda i: numpy_rolling(cap[:,i], 252*5), range(cap.shape[1])))
        roll_yy = pd.DataFrame([[i+1 for i in range(252*5)]]*(basic_data['total_assets'].values.shape[0]-252*5+1)).values.T       
        roll_xy_cov = list(map(lambda i: cov_between_mtx(adj_cap[i].T, roll_yy)*1000000, range(cap.shape[1])))
        slope = np.mat(roll_xy_cov).T/np.mat(adj_cap_var[-(basic_data['total_assets'].values.shape[0]-252*5+1):,:])
        TAG_mtx = -np.vstack((np.zeros((252*5-1, cap.shape[1])), np.mat(slope)))/ma(basic_data['total_assets'], 252*5)
        TAG = pd.DataFrame(TAG_mtx, index = self.stock_return.index, columns = self.stock_return.columns)
        print('TAG因子构建完成，耗时: {} sec'.format(time.time()-tt14))
        '''
        IG因子
        '''  
        tt15 = time.time() 
        cap = basic_data['circulating_cap'].values/1000000
        adj_cap_var = std_roll(basic_data['circulating_cap'].values, 252*5)**2
        adj_cap = list(map(lambda i: numpy_rolling(cap[:,i], 252*5), range(cap.shape[1])))
        roll_yy = pd.DataFrame([[i+1 for i in range(252*5)]]*(basic_data['circulating_cap'].values.shape[0]-252*5+1)).values.T       
        roll_xy_cov = list(map(lambda i: cov_between_mtx(adj_cap[i].T, roll_yy)*1000000, range(cap.shape[1])))
        slope = np.mat(roll_xy_cov).T/np.mat(adj_cap_var[-(basic_data['circulating_cap'].values.shape[0]-252*5+1):,:])
        IG_mtx = -np.vstack((np.zeros((252*5-1, cap.shape[1])), np.mat(slope)))/ma(basic_data['circulating_cap'], 252*5)
        IG = pd.DataFrame(IG_mtx, index = self.stock_return.index, columns = self.stock_return.columns)
        print('IG因子构建完成，耗时: {} sec'.format(time.time()-tt15))

        '''
        CPG因子  financial_expense
        '''  
        tt16 = time.time() 
        cap = basic_data['financial_expense'].values/1000000
        adj_cap_var = std_roll(basic_data['financial_expense'].values, 252*5)**2
        adj_cap = list(map(lambda i: numpy_rolling(cap[:,i], 252*5), range(cap.shape[1])))
        roll_yy = pd.DataFrame([[i+1 for i in range(252*5)]]*(basic_data['financial_expense'].values.shape[0]-252*5+1)).values.T       
        roll_xy_cov = list(map(lambda i: cov_between_mtx(adj_cap[i].T, roll_yy)*1000000, range(cap.shape[1])))
        slope = np.mat(roll_xy_cov).T/np.mat(adj_cap_var[-(basic_data['financial_expense'].values.shape[0]-252*5+1):,:])
        CPG_mtx = -np.vstack((np.zeros((252*5-1, cap.shape[1])), np.mat(slope)))/ma(basic_data['financial_expense'], 252*5)
        CPG = pd.DataFrame(CPG_mtx, index = self.stock_return.index, columns = self.stock_return.columns)  
        print('CPG因子构建完成，耗时: {} sec'.format(time.time()-tt16))
        return DTOA, TOR_VOL, NP_VOL, CEI_VOL, ABS, ACF, ATO, GP, GPM, ROA, TAG, IG, CPG











    # +-------------------------------------------------------+
    # |                        VALUE                          |
    # +-------------------------------------------------------+
    #价值因子，包含STREV(短期反转),SEASON(季节因子),INDMOM(行业动量),RSTR(相对于市场的强度), HIS_ALPHA(历史Alpha)因子
    # def VALUE(self):   