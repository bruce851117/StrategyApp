import numpy as np
import json
with open('./PortfolioApp_data/app_lists.json','r')as f: 
    app_lists = json.load(f)
need_to_reciprocal = app_lists['calculate_need_to_reciprocal']
residual_remove_sign = app_lists['calculate_residual_remove_sign']


def value_transfrom(num,selected_AQ_EM):
    if selected_AQ_EM in need_to_reciprocal:
        num = 1/num
    else:num = num
    measure = (num/abs(num))*np.log(1+abs(num))
    if selected_AQ_EM in residual_remove_sign:
        return abs(measure)
    else:
        return measure