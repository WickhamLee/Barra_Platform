import os
import sys

here_path = os.getcwd()
back1_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
back2_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

sys.path.append(back1_path)


from Basic_code.data_load import basic_data_load
from Basic_code.factor_construct import Industry_factor, Style_factor
# b = basic_data_load()

if __name__ == '__main__':
    a = Style_factor().VOLATILITY()