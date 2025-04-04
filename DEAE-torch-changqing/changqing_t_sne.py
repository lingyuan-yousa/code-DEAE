import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
import seaborn as sns

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', learning_rate=100, perplexity = 35)  #

    x_ts = ts.fit_transform(feat)

    # print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['r','#2D86C0','c','#196F3E','#F4D03E','#AED6F0','m','chocolate','limegreen']
# colors = ['#19703E', '#A56ABE', '#AED6F0', '#2D86C0', '#1A4F71', '#6F2C01', '#DC7733', '#F6B041', '#F4D03E']

#['r','b','c','g','y','y','r','y','r']
#
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }

if __name__ == '__main__':

    all_data = pd.read_csv('changqing.csv', encoding='utf-8')

    x_all_data = all_data.drop(columns=['Well_Name', 'depth', 'RESULT(OG_DZFC)', 'LITH'])
    y_all_data = all_data['LITH'] - 1
    y_all_data = (y_all_data.values).reshape((-1, 1))
    data = visual(x_all_data)
    print(data.shape)
    print(y_all_data.shape)
    data = np.hstack((data,y_all_data))
    data = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1], 'label': data[:, 2]})
    for index in range(9):  # 假设总共有三个类别，类别的表示为0,1,2
        X = data.loc[data['label'] == index]['x']
        Y = data.loc[data['label'] == index]['y']
        # ['r','b','c','g','y','y','m','yellow','yellow']
        if index == 0:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label='T')
        elif index == 1:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label="HgS")
        elif index == 2:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label="DS")
        elif index == 3:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label="SS")
        elif index == 4:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label="HS")
        elif index == 5:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65,label="BS")
        else:
            plt.scatter(X, Y, s=5, c=colors[index], alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        plt.legend(loc='best')
        # plt.legend(loc='upper right')
    #plt.title("name", fontsize=16, fontweight='normal', pad=20)
    #fig = plt.figure(figsize=(32, 32))
    plt.savefig("t-changqing_sne.tif", dpi=400, format="tif")
    plt.show()