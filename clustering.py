from sklearn.cluster import KMeans

def create_km(k=6):
    km=KMeans(n_clusters=k)
    return km