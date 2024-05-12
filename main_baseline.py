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

def find_n_obj_kmeans(km,data_x,num=15,full_data=None,use_dist=False):
    centroids  = km.cluster_centers_
    distances = pairwise_distances(centroids, data_x, metric='euclidean')
    y=10
    f=10
    if use_dist:
        fixed_ind=[]
        cluster_labels=np.array(km.predict(data_x))
        for c in range(len(centroids)):
            dim_min_max={}
            cluster_idx = np.where(cluster_labels == c)[0]
            cluster_size=len(cluster_idx)
            data_x_c=data_x.iloc[cluster_idx]
            dims=data_x.columns
            for dim in dims:
                dim_min_max[dim]=[min(data_x_c[dim])-y,max(data_x_c[dim]+y)]
            rect_data=full_data.loc[(full_data[dims[0]]<dim_min_max[dims[0]][1]) & 
                          (full_data[dims[1]]<dim_min_max[dims[1]][1]) &
                          (full_data[dims[2]]<dim_min_max[dims[2]][1]) &
                          (full_data[dims[3]]<dim_min_max[dims[3]][1]) &
                          (full_data[dims[4]]<dim_min_max[dims[4]][1]) &
                          (dim_min_max[dims[0]][0]<full_data[dims[0]]) &
                          (dim_min_max[dims[1]][0]<full_data[dims[1]]) &
                          (dim_min_max[dims[2]][0]<full_data[dims[2]]) &
                          (dim_min_max[dims[3]][0]<full_data[dims[3]]) &
                          (dim_min_max[dims[4]][0]<full_data[dims[4]])]
            
            # cluster_idx = list(cluster_idx)
            # dist_c=distances[c]
            # distances_cluster=dist_c[cluster_idx]
            # max_dist_c=max(distances_cluster)+y
            # dist_all_c=np.array(distances_all[c])
            # dataset_idx=np.argwhere(dist_all_c<max_dist_c).squeeze(1).tolist()
            fc_data=rect_data.sample(f*cluster_size)
            fixed_ind.extend(fc_data.index)
        
        fixed_ind=np.unique(fixed_ind)
    else:
        ind = [np.argpartition(i, num)[:num] for i in distances]
        fixed_ind=[]
        for i in ind:
            fixed_ind.append(i[0])
    return fixed_ind

def find_n_obj_data(data_x,full_data=None):
    y=10
    f=10
    fixed_ind=[]
    for c in range(len(data_x)):
        dim_min_max={}
        dims=data_x.columns
        for dim in dims:
            dim_min_max[dim]=[data_x.iloc[c][dim]-y,data_x.iloc[c][dim]+y]
        rect_data=full_data.loc[(full_data[dims[0]]<dim_min_max[dims[0]][1]) & 
                        (full_data[dims[1]]<dim_min_max[dims[1]][1]) &
                        (full_data[dims[2]]<dim_min_max[dims[2]][1]) &
                        (full_data[dims[3]]<dim_min_max[dims[3]][1]) &
                        (full_data[dims[4]]<dim_min_max[dims[4]][1]) &
                        (dim_min_max[dims[0]][0]<full_data[dims[0]]) &
                        (dim_min_max[dims[1]][0]<full_data[dims[1]]) &
                        (dim_min_max[dims[2]][0]<full_data[dims[2]]) &
                        (dim_min_max[dims[3]][0]<full_data[dims[3]]) &
                        (dim_min_max[dims[4]][0]<full_data[dims[4]])]
        
        # cluster_idx = list(cluster_idx)
        # dist_c=distances[c]
        # distances_cluster=dist_c[cluster_idx]
        # max_dist_c=max(distances_cluster)+y
        # dist_all_c=np.array(distances_all[c])
        # dataset_idx=np.argwhere(dist_all_c<max_dist_c).squeeze(1).tolist()
        f_data=rect_data.sample(f)
        fixed_ind.extend(f_data.index)
    
    fixed_ind=np.unique(fixed_ind)
    print(len(fixed_ind))
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

def iter_dataset(df,df_all,target_labels):
    labels=[0]
    # while np.count_nonzero(np.array(labels))==0:
    new_df=df.drop(['objid'],axis=1)
    if len(new_df)<6:
        ind=find_n_obj_data(new_df,df_all)
    else:
        km=run_kmeans(new_df,6)
        ind=find_n_obj_kmeans(km,new_df,10,df_all,True)
    
    if len(ind)>20:
        print(ind.shape)
        ind_idx = np.random.choice(ind.shape[0], 20, replace=False)
        ind=ind[ind_idx]
    labels=target_labels[ind]
    dataset=df_all.iloc[ind]
    return dataset,labels

def fit_dc(train_x,train_y):
    dc=DecisionTreeClassifier()
    dc.fit(train_x,train_y)
    return dc


def get_metrics(target_labels,pred_labels):
    return f1_score(target_labels,pred_labels)


user_query="select * from skyphoto where colc > 25 and colc < 30;"

# target_df=utils.run_query_to_df(user_query)
# target_objs=target_df['objid'].values

target_objs=get_target(user_query)

df=pd.read_csv("./data/small_sdss.csv")
df_sampler=pd.read_csv("./data/small_sdss.csv")
all_objid=df['objid'].values
start_df=df.drop(['objid'],axis=1)

target_labels=get_target_labels(target_objs,df)
# dataset,labels=initial_dataset(df,target_labels)

# train_x=dataset.drop(['objid'],axis=1)


sample=20

dataset=df.sample(sample)
idx=dataset.index
labels=target_labels[idx]
dc=fit_dc(dataset.drop(['objid'],axis=1),labels)
pred_labels=dc.predict(start_df)
print("Score :",get_metrics(target_labels,pred_labels))
for i in range(20):
    new_dataset=df.sample(sample)
    idx=new_dataset.index
    new_labels=target_labels[idx]
    dataset=pd.concat([dataset,new_dataset])
    labels=np.concatenate((labels,new_labels))

    print("Length of dataset :",len(dataset))
    new_train_x=dataset.drop(['objid'],axis=1)

    dc=fit_dc(new_train_x,labels)

    pred_labels_full=dc.predict(start_df)
    score=get_metrics(target_labels,pred_labels_full)
    print("Score on full dataset:",score)

    if score==1:
        break
    # pred_labels=np.array(pred_labels)
    # print(np.unique(pred_labels,return_counts=True))
