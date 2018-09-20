import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def cluster_destination():
    #处理原始数据，得到目的地聚类结果及将原始数据转化成20维的7000个样本
    with open("./data_csv/Porto_taxi_data_training.csv",'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    #挑选坐标数据
    coordinate_original=[ eval(i[-1]) for i in rows[1:]]#取出每个样本的坐标轨迹，每条轨迹为[[Lo1,la1],[Lo2,La2],^]
    coordinate=[i for i in coordinate_original if len(i)>6]

    #取每个轨迹的前五个坐标与后五个坐标组成新的轨迹
    coordinate_label=[i[-1] for i in coordinate]#目的地坐标取出
    coordinate_1=[i[:5]+i[-6:-1]for i in coordinate]#坐标取出
    coordinate_2=[sum(i,[]) for i in coordinate_1]#列表扁平化,每个样本20维

    # Incorrect number of clusters
    # y_pred = KMeans(n_clusters=18, random_state=170).fit_predict(coordinate_label)
    net = KMeans(n_clusters=18, random_state=170).fit(coordinate_label[:7000])#聚类18类
    center=net.cluster_centers_#得到聚类中心
    y_pred=net.predict(coordinate_label)
    # coordinate_label= [[i] for i in y_pred]
    print(coordinate_label)

    #坐标归一化
    coordinate_2_np=np.array(coordinate_2)
    # max_coordinate_2_np=np.array([90,180,90,180,90,180,90,180,90,180,90,180,90,180,90,180,90,180,90,180])
    max_coordinate_2_np = np.array([180,90, 180,90,  180,90,  180,90,  180,90,180,90, 180,90, 180,90,180,90,180,90])
    coordinate_2_normal=[np.abs(i/max_coordinate_2_np) for i in coordinate_2_np]
    coordinate_2_normal=np.array(coordinate_2_normal)
    return coordinate_2_normal,y_pred ,np.array(center)
#plt.subplot(221)
#plt.scatter(X[:, 0], X[:, 1], c=y_pred)
#plt.title("Incorrect Number of Blobs")