# coding='utf-8'
"""# 一个对S曲线数据集上进行各种降维的说明。"""
from time import time
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import numpy as np 
from sklearn import manifold, datasets
 
# # Next line to silence pyflakes. This import is needed.
# Axes3D
 
n_points = 1000
# X是一个(1000, 3)的2维数据，color是一个(1000,)的1维数据
X, color = datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2
 
fig = plt.figure(figsize=(8, 8))
# 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)
 
 
'''绘制S曲线的3D图像'''
ax = fig.add_subplot(211, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)  # 初始化视角
 
'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)  # 转换后的输出
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
ax = fig.add_subplot(2, 1, 2)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
 
plt.show()

def awgn(audio, snr):
    #在audio y中 添加噪声 噪声强度SNR为int
    audio_power = audio ** 2 
    audio_average_power = np.mean(audio_power)
    audio_average_db = 10 * np.log10(audio_average_power)
    noise_average_db = audio_average_db - snr
    noise_average_power = 10 ** (noise_average_db / 10)
    mean_noise = 0 
    noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    return audio + noise

