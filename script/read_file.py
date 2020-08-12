# -*- coding: utf-8 -*-
'''
@author : cy023
@date   : 2020.8.9
@brief  : Read dataset filename script.
'''
import os
dirPath = r".\data"

f = open('data_filename.txt', 'w')
for i in range(len(os.listdir(dirPath))):
    f.write(os.listdir(dirPath)[i] + '\n')
    print(os.listdir(dirPath)[i])
f.close()