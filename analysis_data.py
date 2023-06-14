# coding='utf-8'
from time import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import numpy as np 
from sklearn import manifold, datasets
 
# # Next line to silence pyflakes. This import is needed.
# Axes3D
def draw_tsne(X, color,save_path="./picture/t_sne.png"):
    # n_points = 1000
    # X, color = datasets.make_s_curve(n_points, random_state=0)
    n_components = 2
    
    '''t-SNE'''
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    X = np.array(X)
    if len(X.shape)==3:
        X=X.sum(axis=2)/X.shape[2]
    Y = tsne.fit_transform(X) 
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0)) 
    # ax = fig.add_subplot(2, 1, 2)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
 
    plt.savefig(save_path)
    # plt.show()

def draw_tsne_json(filename="res.json",Y_=None):
    with open(filename,"r") as fp:
        X = []
        Y=  []
        for line in fp:
            temp = line.replace("][","]#[").split("#")
            x = json.loads(temp[0])[0]
            X.append(x)
            if Y_ is None:
                y = json.loads(temp[1])
                Y.append(y)
        if Y_  is not None:
            Y=Y_
        draw_tsne(X, Y,f"./picture/{filename.split('.')[0]}" )
    return Y




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


# with open("res.json","r") as fp:
#     X = []
#     Y=  []
#     for line in fp:
#         temp = line.replace("][","]#[").split("#")
#         x = json.loads(temp[0])[0]
#         y = json.loads(temp[1])
#         X.append(x)
#         Y.append(y)
#     draw_tsne(X, Y)
if __name__=="__main__":
    test_YY = draw_tsne_json("res.json")
    draw_tsne_json("data/res_cnn.json",test_YY)
    draw_tsne_json("data/res_first.json",test_YY)
    draw_tsne_json("data/res_informer.json",test_YY)