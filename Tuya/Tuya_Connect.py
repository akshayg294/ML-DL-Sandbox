# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:24:20 2021

@author: Akshay Gupta
"""

import tinytuya
import pandas as pd
main_data=[]
import tinytuya


#%%
d = tinytuya.OutletDevice('32750035500291025a61', '180.151.102.18','2dad4bc4b7a02b16')
d.set_version(3.3)
# d.set_socketPersistent(True)

d.updatedps([18,19,20])
data = d.status() 
main_data.append(pd.Series(data['dps']))

#%%

x=pd.DataFrame(main_data)
x.columns=['Switch 1','Countdown 1','Current','Power','Voltage','Test bit','a','b','c','d']
print(x[['Current','Power','Voltage']])

# d.set_timer(10)
#%%

tinytuya.scan()
1