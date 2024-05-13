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
    initial_indices = []
    idx = 0
    count_ones = 0
    count_zeroes = 0
    while count_ones < 3 or count_zeroes < 3:
        if target_labels[idx] == 1 and count_ones < 3:
            count_ones += 1
            initial_indices.append(idx)
        elif count_zeroes < 3:
            count_zeroes += 1
            initial_indices.append(idx)
        idx += 1
    labels = target_labels[initial_indices]
    dataset = df.iloc[initial_indices]
    #while np.count_nonzero(np.array(labels))!=3:
    #    start_df=df.drop(['objid'],axis=1)
    #    km=run_kmeans(start_df,6)
    #    ind=find_n_obj_kmeans(km,start_df,1)
    #    labels=target_labels[ind]
    #    dataset=df.iloc[ind]
    return dataset,labels

boundary_exploitation_cap = 5

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
    if len(ind)<20:
        remaining_samples = 20 - len(ind)
        regions = get_regions(last_trained_x, last_predicted_y)
        boundary_df = pd.DataFrame()
        expanded_dfs = []
        min_length = None
        for region in regions:
            expanded_df = expand_region(region, df_all)
            if len(expanded_df.index) != 0:
                expanded_dfs.append(expanded_df)
                if min_length is None:
                    min_length = len(expanded_df.index)
                if min_length > len(expanded_df.index):
                    min_length = len(expanded_df.index)
        for expanded_df in expanded_dfs:
            expanded_df = expanded_df.sample(min_length)
            boundary_df = pd.concat([boundary_df, expanded_df])
        if len(boundary_df.index) > remaining_samples:
            boundary_df = boundary_df.sample(remaining_samples)
        global boundary_exploitation_cap
        if len(boundary_df.index) > boundary_exploitation_cap:
            boundary_df = boundary_df.sample(int(boundary_exploitation_cap))
        dataset = pd.concat([dataset, boundary_df])
        if len(dataset.index) < 20:
            remaining_samples = 20 - len(dataset.index)
            dataset = pd.concat([dataset, df_all.sample(remaining_samples)])
        ind = dataset.index
        labels = target_labels[ind]
    return dataset,labels

last_trained_x = pd.DataFrame()
last_predicted_y = []

def fit_dc(train_x,train_y):
    dc=DecisionTreeClassifier()
    dc.fit(train_x,train_y)
    global last_trained_x
    global last_predicted_y
    last_trained_x = train_x
    last_predicted_y = dc.predict(train_x)
    return dc


def get_metrics(target_labels,pred_labels):
    return f1_score(target_labels,pred_labels)


def get_regions(training_data = last_trained_x, predicted_labels = last_predicted_y):
    negative_example_indices = []
    positive_example_indices = []
    for idx in range(len(predicted_labels)):
        if predicted_labels[idx] == 0:
            negative_example_indices.append(idx)
        if predicted_labels[idx] == 1:
            positive_example_indices.append(idx)
    negative_examples = training_data.iloc[negative_example_indices]
    positive_examples = training_data.iloc[positive_example_indices]
    features = [col for col in positive_examples.columns]
    region = dict()
    for feature in features:
        region[feature] = (positive_examples[feature].min(),
                           positive_examples[feature].max())
    split_point = None
    for feature in features:
        min_bound, max_bound = region[feature]
        for idx, negative_example in negative_examples.iterrows():
            if negative_example[feature] >= min_bound and \
                negative_example[feature] <= max_bound:
                split_point = negative_example
                break
        if split_point is not None:
            break
    if split_point is None:
        return [region]
    groups = []
    for bitmask in range(1<<(len(features))):
        filtered_df = positive_examples
        for i in range(len(features)):
            if len(filtered_df.index) == 0:
                break
            if (bitmask & (1<<i)) != 0:
                filtered_df = filtered_df[(filtered_df[features[i]] < split_point[features[i]])]
            else:
                filtered_df = filtered_df[(filtered_df[features[i]] > split_point[features[i]])]
        if len(filtered_df.index) != 0:
            groups.append(filtered_df)

    regions = []
    for group in groups:
        group_training_x = pd.concat([group, negative_examples])
        group_predicted_labels = []
        for _ in range(len(group.index)):
            group_predicted_labels.append(1)
        for _ in range(len(negative_examples.index)):
            group_predicted_labels.append(0)
        sub_regions = get_regions(group_training_x, group_predicted_labels)
        for region in sub_regions:
            regions.append(region)
    return regions


def expand_region(region, df):
    x = 3
    #features = list(region.keys())
    features = ['colc']
    for bitmask in range(1<<len(features)):
        filtered_df = df
        for i in range(len(features)):
            if len(filtered_df.index) == 0:
                break
            min_bound, max_bound = region[features[i]]
            if (bitmask & (1<<i)) != 0:
                filtered_df = filtered_df[((filtered_df[features[i]] >= (min_bound - x)) & 
                                           (filtered_df[features[i]] <= (min_bound + x))) |
                                          ((filtered_df[features[i]] >= (max_bound - x)) &
                                           (filtered_df[features[i]] <= (max_bound + x)))]
            else:
                filtered_df = filtered_df[((filtered_df[features[i]] >= min_bound) &
                                           (filtered_df[features[i]] <= max_bound))]
    return filtered_df


user_query="select * from skyphoto where colc >= 25 and colc <= 28;"

# target_df=utils.run_query_to_df(user_query)
# target_objs=target_df['objid'].values

target_objs=get_target(user_query)

df=pd.read_csv("./data/small_sdss.csv")
df_sampler=pd.read_csv("./data/small_sdss.csv")
all_objid=df['objid'].values
start_df=df.drop(['objid'],axis=1)

target_labels=get_target_labels(target_objs,df)
dataset,labels=initial_dataset(df,target_labels)

train_x=dataset.drop(['objid'],axis=1)
dc=fit_dc(train_x,labels)

number_of_samples=20

training_x=df.sample(number_of_samples)
idx=training_x.index
training_y=target_labels[idx]
pred_labels=dc.predict(training_x.drop(['objid'],axis=1))

print("Score :",get_metrics(training_y,pred_labels))
dataset=pd.concat([dataset,training_x])
labels=np.concatenate((labels,training_y))
score = 0.0
last_score = 0.0
while score <= 0.95:
    unq = np.array([x + 2*y for x, y in zip(pred_labels, labels)])
    fn = np.array(np.where(unq == 2)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]
    fn_fp=fn+fp
    new_dataset=dataset.iloc[fn_fp]
    new_labels=labels[fn_fp]

    print("Misclassified data :",len(new_dataset))
    print("Misclassified labels :",len(new_labels))
    # if len(new_dataset)<6:
    #     break
    if len(new_dataset)==0:
        regions = get_regions(last_trained_x, last_predicted_y)
        boundary_df = pd.DataFrame()
        expanded_dfs = []
        min_length = None
        for region in regions:
            expanded_df = expand_region(region, df)
            if len(expanded_df.index) != 0:
                expanded_dfs.append(expanded_df)
                if min_length is None:
                    min_length = len(expanded_df.index)
                if min_length > len(expanded_df.index):
                    min_length = len(expanded_df.index)
        for expanded_df in expanded_dfs:
            expanded_df = expanded_df.sample(min_length)
            boundary_df = pd.concat([boundary_df, expanded_df])

        if len(boundary_df.index) > number_of_samples:
            boundary_df = boundary_df.sample(number_of_samples)
        if len(boundary_df.index) > boundary_exploitation_cap:
            boundary_df = boundary_df.sample(int(boundary_exploitation_cap))
        training_x = boundary_df
        if len(training_x.index) < number_of_samples:
            remaining_samples = number_of_samples - len(training_x.index)
            training_x= pd.concat([training_x, df.sample(remaining_samples)])
        idx=training_x.index
        training_y=target_labels[idx]
        dataset=pd.concat([dataset,training_x])
        labels=np.concatenate((labels,training_y))
        pred_labels=dc.predict(dataset.drop(['objid'],axis=1))
        # print("OG LABELS",labels)
        # print("PRED LABELS",pred_labels)

        dc=fit_dc(dataset.drop(['objid'],axis=1),labels)

        pred_labels_full=dc.predict(start_df)
        score=get_metrics(target_labels,pred_labels_full)
        print("Length of dataset :",len(dataset))
        print("Score on full dataset:",score)
        pred_labels=dc.predict(dataset.drop(['objid'],axis=1))
        continue
    new_dataset,new_labels=iter_dataset(new_dataset,start_df,target_labels)

    dataset=pd.concat([dataset,new_dataset])
    labels=np.concatenate((labels,new_labels))

    # dataset['label']=labels
    # no_dup_dataset=dataset.drop_duplicates(subset=['objid'])
    # labels=no_dup_dataset['label'].values
    # dataset=no_dup_dataset.drop(['label'],axis=1)

    print("Length of dataset :",len(dataset))
    new_train_x=dataset.drop(['objid'],axis=1)

    dc=fit_dc(new_train_x,labels)

    pred_labels_full=dc.predict(start_df)
    score=get_metrics(target_labels,pred_labels_full)
    print("Score on full dataset:",score)

    pred_labels=dc.predict(dataset.drop(['objid'],axis=1))
    boundary_exploitation_cap += 0.1
    if score - last_score < 0.01:
        boundary_exploitation_cap -= 1
    last_score = score

    # pred_labels=np.array(pred_labels)
    # print(np.unique(pred_labels,return_counts=True))
