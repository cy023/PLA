import os
dirPath = r".\data"

f = open('data_filename.txt', 'w')
for i in range(len(os.listdir(dirPath))):
    f.write(os.listdir(dirPath)[i] + '\n')
    print(os.listdir(dirPath)[i])
f.close()