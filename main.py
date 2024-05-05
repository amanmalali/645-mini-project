import utils
import pandas as pd
from scipy.cluster.vq import vq
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances,f1_score
from sklearn.cluster import KMeans
from clustering import create_km


'''
TODO:
1. Run user query and get the obj id to know which ones are relevant
2. Normalize dataset to 0-100 range for every column
3. Start by clustering the data, k=6? higher level clustering
4. Pick center of clusters, assign positive or negative based on user query
5. Train decision tree classifier on this data
6. Predict on dataset
7. Run k means on misclassified points and select 10-15 points from each cluster
9. Do the boundry exploration bit
'''

# utils.select_columns("./data/sdss_100k.csv")
# utils.init_tables()

def get_target(user_query):
    target_df=utils.run_query_to_df(user_query)
    target_objs=target_df['objid'].values
    return target_objs

def get_target_labels(target_objs,df):
    all_objid=df['objid'].values   
    target_labels=[]

    for obj in all_objid:
        if obj in target_objs:
            target_labels.append(1)
        else:
            target_labels.append(0)
    
    return np.array(target_labels)


def run_kmeans(data_df,k=6):
    km=KMeans(n_clusters=k)
    labels=km.fit_predict(data_df)
    return km

def find_n_obj_kmeans(km,data_x,num=15,use_dist=False):
    centroids  = km.cluster_centers_
    distances = pairwise_distances(centroids, data_x, metric='euclidean')
    ind = [np.argpartition(i, num)[:num] for i in distances]
    y=-10
    if use_dist:
        fixed_ind=[]
        cluster_labels=np.array(km.predict(data_x))
        for c in range(len(centroids)):
            cluster_idx = np.where(cluster_labels == c)[0]
            cluster_idx = list(cluster_idx)
            dist_c=distances[c]
            distances_cluster=dist_c[cluster_idx]
            max_dist_c=max(distances_cluster)+y
            dist_c=np.array(dist_c)
            dataset_idx=np.argwhere(dist_c<max_dist_c).squeeze(1).tolist()
            fixed_ind.extend(dataset_idx)
        
        fixed_ind=np.unique(fixed_ind)
        print(len(fixed_ind))
    else:
        fixed_ind=[]
        for i in ind:
            fixed_ind.append(i[0])
    return fixed_ind


def initial_dataset(df,target_labels):
    labels=[0,0,0,0,0]
    while np.count_nonzero(np.array(labels))==0 or np.count_nonzero(np.array(labels))==5:
        start_df=df.drop(['objid'],axis=1)
        km=run_kmeans(start_df,6)
        ind=find_n_obj_kmeans(km,start_df,1)
        labels=target_labels[ind]
        dataset=df.iloc[ind]
    return dataset,labels

def iter_dataset(df,target_labels):
    labels=[0]
    while np.count_nonzero(np.array(labels))==0:
        new_df=df.drop(['objid'],axis=1)
        km=run_kmeans(new_df,6)
        ind=find_n_obj_kmeans(km,new_df,10,True)
        labels=target_labels[ind]
        dataset=df.iloc[ind]
    return dataset,labels

def fit_dc(train_x,train_y):
    dc=DecisionTreeClassifier()
    dc.fit(train_x,train_y)
    return dc


def get_metrics(target_labels,pred_labels):
    return f1_score(target_labels,pred_labels)


user_query="select * from skyphoto where colc > 25 and colc < 75 and rowc > 20 and rowc < 80;"

# target_df=utils.run_query_to_df(user_query)
# target_objs=target_df['objid'].values

target_objs=get_target(user_query)

df=pd.read_csv("./data/small_sdss.csv")
all_objid=df['objid'].values
start_df=df.drop(['objid'],axis=1)

target_labels=get_target_labels(target_objs,df)
dataset,labels=initial_dataset(df,target_labels)


train_x=dataset.drop(['objid'],axis=1)
dc=fit_dc(train_x,labels)

pred_labels=dc.predict(start_df)

print("Score :",get_metrics(target_labels,pred_labels))


for i in range(10):
    unq = np.array([x + 2*y for x, y in zip(pred_labels, target_labels)])
    fn = np.array(np.where(unq == 2)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]

    fn_fp=fn+fp

    new_dataset=df.iloc[fn_fp]
    new_labels=target_labels[fn_fp]

    print("Misclassified data :",len(new_dataset))
    print("Misclassified labels :",len(new_labels))

    new_dataset,new_labels=iter_dataset(new_dataset,new_labels)

    dataset=pd.concat([dataset,new_dataset])
    labels=np.concatenate((labels,new_labels))

    new_train_x=dataset.drop(['objid'],axis=1)

    dc=fit_dc(new_train_x,labels)

    pred_labels=dc.predict(start_df)
    score=get_metrics(target_labels,pred_labels)
    print("Score :",get_metrics(target_labels,pred_labels))

    print(pred_labels)
    if score==1:
        break
    # pred_labels=np.array(pred_labels)
    # print(np.unique(pred_labels,return_counts=True))
