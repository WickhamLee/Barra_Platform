import yaml
import os
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
here_path = os.getcwd()
back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# +-------------------------------------------------------+
# |                     config_load                       |
# +-------------------------------------------------------+
#把写有数据库路径的yaml文件读取进来
def config_load():
    global here_path, back1_path, back2_path
    file = open(os.path.join(back1_path, 'Config', 'config_.yml'), 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # 指定Loader
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data

# +-------------------------------------------------------+
# |                    data_load                          |
# +-------------------------------------------------------+
#把需要的底层数据读取进来    
def basic_data_load():
    config_ = config_load()
    
    data_must_have = config_['Data_Must_Need'] + config_['Industry_Class']
    data_warehouse = config_['Data_Warehouse_Path']
    
    out_put_data = {}
    for i in data_must_have:
        try:
            out_put_data[i] = pd.read_hdf(os.path.join(data_warehouse, 'Backtest', 'Ashares', 'All', '1day', i+'.hd5'), 
                                      key = 'table')
            print(os.path.join(data_warehouse, 'Backtest', 'Ashares', 'All', '1day', i+'.hd5') + '已载入')
        except:
            out_put_data[i] = pd.read_hdf(os.path.join(data_warehouse, 'Backtest', 'Ashares_indices', 'market_data', i+'.csv'))
            print(os.path.join(data_warehouse, 'Backtest', 'Ashares_indices', 'market_data', i+'.csv') + '已载入')
    out_put_data[config_['Independent_Variable'][0]] = pd.read_csv(os.path.join(data_warehouse, 
                                                                       'Backtest', 
                                                                       'Ashares_indices', 
                                                                       'market_data',
                                                                       config_['Independent_Variable'][0]+'.csv'),
                                                                   index_col = [0]
                                                          )
    print(os.path.join(data_warehouse,'Backtest', 'Ashares_indices', 'market_data', config_['Independent_Variable'][0]+'.csv') + '已载入')
    out_put_data['in_' + config_['Independent_Variable'][0]] = pd.read_hdf(os.path.join(data_warehouse, 
                                                                       'Backtest', 
                                                                       'Ashares', 
                                                                       'All',
                                                                       '1day',
                                                                       'in_' + config_['Independent_Variable'][0]+'.hd5'),
                                                                   key = 'table'
                                                          )
    print(os.path.join(data_warehouse, 'Backtest', 'Ashares', 'All', '1day', 'in_' + config_['Independent_Variable'][0]+'.hd5') +'已载入')
    out_put_data['trade_date'] = pd.read_csv(os.path.join(data_warehouse,'Backtest', 'Ashares', 'All', '1day', 'trade_date.csv'),
                                             index_col = [0])
    print(os.path.join(data_warehouse,'Backtest', 'Ashares', 'All', '1day', 'trade_date.csv') + '已载入')

    # print('开始进行数据行对齐')
    st_ = config_['Analyse_Start_date']
    ed_ = config_['Analyse_End_date']
    
    while st_ not in list(out_put_data['trade_date']['date']):
        st_ = (dt.datetime.strptime(st_, '%Y-%m-%d') + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    while ed_ not in list(out_put_data['trade_date']['date']):
        ed_ = (dt.datetime.strptime(ed_, '%Y-%m-%d') - dt.timedelta(days=1)).strftime("%Y-%m-%d")

    for i in tqdm(out_put_data, desc = "开始进行数据行对齐"):
        out_put_data[i] = out_put_data[i].loc[st_:ed_, :]
    
    # print(config_['Independent_Variable'])
    standard_stock_list = list(out_put_data['close'].columns)
    for i in tqdm(out_put_data, desc = "开始进行数据列对齐"):
        if (i not in config_['Independent_Variable']) & (i != 'trade_date'):
            lack_list = list(set(standard_stock_list).difference(set(list(out_put_data[i].columns))))
            for j in lack_list:
                out_put_data[i][j] = [np.nan]*out_put_data[i].shape[0]
                out_put_data[i] = (out_put_data[i].T).sort_index().T
                # out_put_data[i].to_hdf('E:\\BarraPlatform\\Barra_Recurrent\\Temporary_warehouse\\'+i+'.hd5', key='table')
    # print('数据对齐完成')
    return out_put_data