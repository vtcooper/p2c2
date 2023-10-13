import numpy as np

def haversine_pair(u1,v1):
    ## u1 and v1 are both row vectors [lat,lon] in radians
    ## result in km based on earth radius
    hdist = 6371000/1000 * 2*np.arcsin(np.sqrt(
        np.sin((u1[0]-v1[0])/2)**2 + 
        np.sin((u1[1]-v1[1])/2)**2 * np.cos(u1[0]) * np.cos(v1[0])))
    return hdist