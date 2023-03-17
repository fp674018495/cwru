#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def draw_matrx(y_test,y_pred):
    save_flg = True
    confusion = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 5))
    
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.colorbar() 
    indices = range(len(confusion))    
    classes = list(range(10))
    plt.xticks(indices, classes, rotation=45)
    plt.yticks(indices, classes)
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  

    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.
    
    for i in range(len(confusion)):    
        for j in range(len(confusion[i])): 
            plt.text(j, i, format(confusion[i][j], fmt),
            fontsize=16,  
            horizontalalignment="center", 
            verticalalignment="center",  
            color="white" if confusion[i, j] > thresh else "black")

    if save_flg:  
        plt.savefig("./picture/confusion_matrix.png")
    
    
    # 7.显示
    plt.show()