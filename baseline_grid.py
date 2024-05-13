import utils
import pandas as pd
from scipy.cluster.vq import vq
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances,f1_score
from sklearn.cluster import KMeans
from clustering import create_km
from itertools import product
from scipy.spatial.distance import cdist




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

def fit_dc(train_x,train_y):
    dc=DecisionTreeClassifier()
    dc.fit(train_x,train_y)
    return dc


def get_metrics(target_labels,pred_labels):
    return f1_score(target_labels,pred_labels)

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return np.argmin(cdist([point], points)[0],0)


user_query="select * from skyphoto where colc >= 9 and colc <= 12;"

# target_df=utils.run_query_to_df(user_query)
# target_objs=target_df['objid'].values

target_objs=get_target(user_query)

df=pd.read_csv("./data/small_sdss.csv")
df_sampler=pd.read_csv("./data/small_sdss.csv")
all_objid=df['objid'].values
start_df=df.drop(['objid'],axis=1)

target_labels=get_target_labels(target_objs,df)

dim_range={}
dims=start_df.columns
for dim in dims:
    dim_range[dim]=[min(start_df[dim]),(max(start_df[dim])-min(start_df[dim]))/2,max(start_df[dim])]


grid_centers={}


for dim in dims:
    grid_centers[dim]=[(dim_range[dim][1]+dim_range[dim][0])/2,(dim_range[dim][2]+dim_range[dim][1])/2]

center_lists=[]
for dim in dims:
    center_lists.append(grid_centers[dim])


centers_to_search=[]
for item in product(*center_lists):
    centers_to_search.append(list(item))


all_grid_centers={}

all_grid_centers[0]=[25,75]
all_grid_centers[1]=[12.5,37.5,62.5,87.5]
all_grid_centers[2]=[6.25,18.75,31.25,43.75,56.25,68.75,81.25,93.75]

centers_to_search=[]
for i in range(3):
    center_lists=[]
    for dim in dims:
        center_lists.append(all_grid_centers[i])
    for item in product(*center_lists):
        centers_to_search.append(list(item))


searched_val=[0]*len(centers_to_search)
idx=[]
num_val=0
for i in range(len(centers_to_search)):
    if searched_val[i]==0:
        grid_i=centers_to_search[i]
        close_idx=closest_point(grid_i,start_df.values)
        idx.append(close_idx)
        searched_val[i]=1
        num_val=num_val+1
        if num_val>=20:
            break


dataset=df.iloc[idx]
labels=target_labels[idx]
dc=fit_dc(dataset.drop(['objid'],axis=1),labels)
pred_labels=dc.predict(start_df)
score=get_metrics(target_labels,pred_labels)
print("Score :",score)

while score<=0.95:
    idx=[]
    num_val=0
    for i in range(len(centers_to_search)):
        if searched_val[i]==0:
            grid_i=centers_to_search[i]
            close_idx=closest_point(grid_i,start_df.values)
            idx.append(close_idx)
            searched_val[i]=1
            num_val=num_val+1
            if num_val>=20:
                break

# print(searched_val)
    new_data=df.iloc[idx]
    new_labels=target_labels[idx]

    dataset=pd.concat([dataset,new_data],axis=0)
    labels=np.concatenate([labels,new_labels],axis=0)
    dc=fit_dc(dataset.drop(['objid'],axis=1),labels)
    pred_labels=dc.predict(start_df)
    score=get_metrics(target_labels,pred_labels)
print("Score :",score)
print("DATASET LENGTH",len(dataset))
# print("Score :",get_metrics(target_labels,pred_labels))



