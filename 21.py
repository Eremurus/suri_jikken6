import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#データ読み込み
df = pd.read_csv("./suri_jikken6_data/mmse_kadai14.txt",header=None)
data = np.array(df)

x_1 = np.array(data[:,0])
x_2 = np.array(data[:,1])
best_dist = 10**6

K = 3
N = 1000
repeat_times = 10
f_ = open("21_meu.txt","w")

#プロットの準備
fig = plt.figure()
ax1 = fig.add_subplot(3, 4, 1)
ax2 = fig.add_subplot(3, 4, 2)
ax3 = fig.add_subplot(3, 4, 3)
ax4 = fig.add_subplot(3, 4, 4)
ax5 = fig.add_subplot(3, 4, 5)
ax6 = fig.add_subplot(3, 4, 6)
ax7 = fig.add_subplot(3, 4, 7)
ax8 = fig.add_subplot(3, 4, 8)
ax9 = fig.add_subplot(3, 4, 9)
ax10 = fig.add_subplot(3, 4, 10)
ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

#10 組みの初期値に対して同様の作業を行う
for time in range(repeat_times):
    x1 = np.random.rand()*20.0 - 10.0
    x2 = np.random.rand()*20.0 - 10.0
    x3 = np.random.rand()*20.0 - 10.0
    y1 = np.random.rand()*10.0 - 5.0
    y2 = np.random.rand()*10.0 - 5.0
    y3 = np.random.rand()*10.0 - 5.0
    meu = [[x1, y1],[x2, y2],[x3 ,y3]]
    meu_tmp = meu
    f_.writelines([str(meu),"\n"])
    meu_pre = [[0.0, 0.0] for _ in range(K)]

    meu = np.array(meu)
    meu_pre = np.array(meu_pre)

    j = 0
    epsilon = 10 ** (-6)

    tmp = np.sum((meu_pre-meu)**2,axis=1)
    max_movement = tmp.max()#パラメータの反復後の変動

    #k-means
    while epsilon <= max_movement:
        r = np.array([[0 for l in range(K)] for i in range(N)])
        j += 1
        #それぞれのデータに対して
        for i in range(N):
            l_one = 0
            dist_min = 100000000
            x = np.array([x_1[i], x_2[i]])
            #クラスを決定
            for l in range(K):
                if np.sqrt(np.sum((meu[l]-x)**2)) < dist_min:
                    l_one = l
                    dist_min = np.sqrt(np.sum((meu[l]-x)**2))
            #そのデータに割り振られたクラスのindex が1
            r[i][l_one] = 1
        #それぞれのクラスについて
        for l in range(K):
            bunbo = np.sum(r, axis=0)[l]
            bunsi = np.array([0.0, 0.0])
            for i in range(N):
                x = np.array([x_1[i], x_2[i]])
                bunsi += r[i][l] * x
            meu_pre[l] = meu[l]
            if bunbo != 0:
                meu[l] = bunsi / bunbo
        tmp = np.sum((meu_pre-meu)**2,axis=1)
        #前の反復からの変動幅
        max_movement = tmp.max()

    #クラス
    clusters = np.where(r == 1)

    #クラスタ0 に分類されたデータ
    cluster_0 = np.where(clusters[1] == 0)
    #クラスタ0 に分類されたデータの第一成分
    cluster_0_1 = [x_1[i] for i in cluster_0[0]]
    #クラスタ0 に分類されたデータの第二成分
    cluster_0_2 = [x_2[i] for i in cluster_0[0]]

    cluster_1 = np.where(clusters[1] == 1)
    cluster_1_1 = [x_1[i] for i in cluster_1[0]]
    cluster_1_2 = [x_2[i] for i in cluster_1[0]]

    cluster_2 = np.where(clusters[1] == 2)
    cluster_2_1 = [x_1[i] for i in cluster_2[0]]
    cluster_2_2 = [x_2[i] for i in cluster_2[0]]

    #プロット
    ax = ax_list[time]
    ax.scatter(cluster_0_1, cluster_0_2)
    ax.scatter(cluster_1_1, cluster_1_2)
    ax.scatter(cluster_2_1, cluster_2_2)

    dist_sum = 0.0 #各データが所属するクラスタの重心からの平均距離
    a = np.array([[cluster_0_1[i], cluster_0_2[i]] for i in range(len(cluster_0_2))])
    b = np.array(meu[0])
    if len(cluster_0_2) != 0:
        dist_sum += np.sum(np.sqrt(np.sum((a - b)**2, axis=1)))

    a = np.array([[cluster_1_1[i], cluster_1_2[i]] for i in range(len(cluster_1_2))])
    b = np.array(meu[1])
    if len(cluster_1_2) != 0:
        dist_sum += np.sum(np.sqrt(np.sum((a - b)**2, axis=1)))

    a = np.array([[cluster_2_1[i], cluster_2_2[i]] for i in range(len(cluster_2_2))])
    b = np.array(meu[2])
    if len(cluster_2_2) != 0:
        dist_sum += np.sum(np.sqrt(np.sum((a - b)**2, axis=1)))

    dist_sum = dist_sum / 1000.0

    #今までの初期値の中で最も良い分類だった場合
    if dist_sum < best_dist:
        best_cluster_0_1 = cluster_0_1
        best_cluster_0_2 = cluster_0_2
        best_cluster_1_1 = cluster_1_1
        best_cluster_1_2 = cluster_1_2
        best_cluster_2_1 = cluster_2_1
        best_cluster_2_2 = cluster_2_2
        best_dist = dist_sum
        meu_best = meu_tmp

#初期値全種類の場合についてプロット
plt.show()

plt.scatter(best_cluster_0_1, best_cluster_0_2)
plt.scatter(best_cluster_1_1, best_cluster_1_2)
plt.scatter(best_cluster_2_1, best_cluster_2_2)

#最も良い分類をプロット
plt.show()

#出力
f_.writelines(["最も良いクラスタを与えた初期値は",str(meu_best)])
f = open("21_clusters.txt","w")
f.write("各データ点の、所属するクラスタの重心からの距離の平均は以下 \n")
f.write(str(best_dist))
f.write("\n")
f.write("cluster0 は以下の点 \n")
f.writelines([str((best_cluster_0_1[i], best_cluster_0_2[i])) for i in range(len(best_cluster_0_2))])
f.write("\n")
f.write("cluster1 は以下の点 \n")
f.writelines([str((best_cluster_1_1[i], best_cluster_1_2[i])) for i in range(len(best_cluster_1_2))])
f.write("\n")
f.write("cluster2 は以下の点 \n")
f.writelines([str((best_cluster_2_1[i], best_cluster_2_2[i])) for i in range(len(best_cluster_2_2))])

f.close()
f_.close()