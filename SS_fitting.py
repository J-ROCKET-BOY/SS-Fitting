import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#球面フィッティング
def SS_fit(data) : 
    #x,y,z要素取り出し
    x = data[:,[0]]
    y = data[:,[1]]
    z = data[:,[2]]

    #データの長さを格納（n = Σ1)    
    n = len(x)
    
    #それぞれの要素の二乗を求める
    x2 = np.power(x,2)
    y2 = np.power(y,2)
    z2 = np.power(z,2)


    #違う要素との積を求める
    xy = x*y
    xz = x*z
    yz = y*z

    #右辺の数値用
    E = -x*(x2+y2+z2)
    F = -y*(x2+y2+z2)
    G = -z*(x2+y2+z2)
    H =   -(x2+y2+z2)
    


    #要素の総和に変換
    x = np.sum(x)
    y = np.sum(y)
    z = np.sum(z)

    x2 = np.sum(x2)
    y2 = np.sum(y2)
    z2 = np.sum(z2)

    xy = np.sum(xy)
    xz = np.sum(xz)
    yz = np.sum(yz)    

    E = np.sum(E)
    F = np.sum(F)
    G = np.sum(G)
    H = np.sum(H)

    #左辺の4×4行列を作る
    K = np.array([  [x2,xy,xz,x],
                    [xy,y2,yz,y],
                    [xz,yz,z2,z],
                    [x,y,z,n]])
    
    #右辺の4×1行列を作る
    L = np.array([E,F,G,H])

    #A,B,C,Dの行列を計算
    P = np.dot(np.linalg.inv(K),L)

    A = P[0]
    B = P[1]
    C = P[2]
    D = P[3]

    #中心座標と半径に変換
    x0 = (-1/2)* A
    y0 = (-1/2)* B
    z0 = (-1/2)* C
    r  = pow(pow(x0,2)+pow(y0,2)+pow(z0,2)-D,1/2)
    
    return np.array([x0,y0,z0,r])



#メイン

#csvファイル読み込み
Mag = np.loadtxt("MagData.csv", delimiter=',')

#x,y,z軸分離
MagX = Mag[:,[0]]
MagY = Mag[:,[1]]
MagZ = Mag[:,[2]]

#3Dプロット
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(MagX, MagY, MagZ)
plt.show()

#球面フィッティング
S = SS_fit(Mag)

#球面フィッティング後の4変数を表示
print(S)

#中心を0に移動
MagX = MagX - S[0]
MagY = MagY - S[1]
MagZ = MagZ - S[2]

#3Dプロット(中心0)
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(MagX, MagY, MagZ)
plt.show()
