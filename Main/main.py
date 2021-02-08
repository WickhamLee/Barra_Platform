import os
import sys

here_path = os.getcwd()
back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

sys.path.append(back1_path)


# from Basic_code.data_load import basic_data_load
from Basic_code.factor_construct import Industry_factor, Style_factor

if __name__ == '__main__':
    # b = basic_data_load()

    SF = Style_factor()
    fac_LNCAP, fac_MIDCAP = SF.SIZE()
    fac_BETA, fac_HIST_SIGMA, fac_DAILY_STD, fac_CUMULATIVE_RANGE = SF.VOLATILITY()
    fac_STOM, fac_STOQ, fac_STOA, fac_SUMSTOM = SF.LIQUIDITY()
    fac_STREV, fac_SEASON, fac_INDMOM, fac_RSTR, fac_HIS_ALPHA = SF.MOMENTUM()
    
    (fac_DTOA, fac_TOR_VOL, fac_NP_VOL, fac_CEI_VOL, fac_ABS, fac_ACF, fac_ATO, 
        fac_GP, fac_GPM, fac_ROA, fac_TAG, fac_IG, fac_CPG) = SF.QUALITY()
    
    # stock_ret = SF.stock_return
    # fac_BETA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_BETA.hd5', key='table')
    # fac_HIST_SIGMA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_HIST_SIGMA.hd5', key='table')
    # fac_DAILY_STD.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_DAILY_STD.hd5', key='table')
    # fac_CUMULATIVE_RANGE.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_CUMULATIVE_RANGE.hd5', key='table')
    # fac_STOM.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_STOM.hd5', key='table')
    # fac_STOQ.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_STOQ.hd5', key='table')
    # fac_STOA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_STOA.hd5', key='table')
    # fac_SUMSTOM.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_SUMSTOM.hd5', key='table')
    # fac_STREV.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_STREV.hd5', key='table')
    # fac_SEASON.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_SEASON.hd5', key='table')
    # fac_INDMOM.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_INDMOM.hd5', key='table')
    # fac_RSTR.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_RSTR.hd5', key='table')
    # fac_HIS_ALPHA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_HIS_ALPHA.hd5', key='table')
    # fac_DTOA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_DTOA.hd5', key='table')
    # fac_TOR_VOL.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_TOR_VOL.hd5', key='table')
    # fac_NP_VOL.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_NP_VOL.hd5', key='table')
    # fac_CEI_VOL.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_CEI_VOL.hd5', key='table')
    # fac_ABS.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_ABS.hd5', key='table')
    # fac_ACF.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_ACF.hd5', key='table')
    # fac_ATO.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_ATO.hd5', key='table')
    # fac_GP.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_GP.hd5', key='table')
    # fac_GPM.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_GPM.hd5', key='table')
    # fac_ROA.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_ROA.hd5', key='table')
    # fac_TAG.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_TAG.hd5', key='table')
    # fac_IG.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_IG.hd5', key='table')
    # fac_CPG.to_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_CPG.hd5', key='table')
    
    # import pandas as pd
    # # # fac_BETA=pd.read_hdf(r'E:\BarraPlatform\Barra_Recurrent\Temporary_warehouse\factor_tem_whouse\fac_BETA.hd5', key='table')
    
    # fac_list = {'fac_BETA':1, 'fac_HIST_SIGMA':1, 'fac_DAILY_STD':1, 'fac_CUMULATIVE_RANGE':1, 'fac_STOM':1, 'fac_STOQ':1, 
    #             'fac_STOA':1, 'fac_SUMSTOM':1, 'fac_STREV':1, 'fac_SEASON':1, 'fac_INDMOM':1, 'fac_RSTR':1, 'fac_HIS_ALPHA':1, 
    #             'fac_DTOA':1, 'fac_TOR_VOL':1, 'fac_NP_VOL':1, 'fac_CEI_VOL':1, 'fac_ABS':1, 'fac_ACF':1, 'fac_ATO':1, 
    #             'fac_GP':1, 'fac_GPM':1, 'fac_ROA':1, 'fac_TAG':1, 'fac_IG':1, 'fac_CPG':1}
    
    # for i in fac_list:
    #     exec(i + "=pd.read_hdf(r'E:\\BarraPlatform\\Barra_Recurrent\\Temporary_warehouse\\factor_tem_whouse\\"+i+".hd5', key='table')")
        
    # for i in range(100):
    #     loc=i-100
    #     dic_x = {}
    #     for j in fac_list:
    #         print("dic_x['"+j+"']=fac_list["+j+"].values["+str(loc)+"]")
    #         exec("dic_x['"+j+"']=fac_list["+j+"].values["+str(loc)+"]")