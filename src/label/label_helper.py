import pandas as pd
import numpy as np
import urllib.request

def get_image(ID, df):
    output = "Images"
    if not os.path.exists(output):
        os.mkdir(output)
    url = df[df.asin==ID][output].values[0][0]
    r = urllib.request.urlopen(url)
    with open(f'{output}/{ID}.jpg', 'wb') as handler:
        handler.write(r.read())

def get_matches(num, connected_components):
    """Returns a list of tuples of a matched pair of products. A pair of products will be considered as matched
    if they are in the same connected components.

    Parameters:
    num (int): Number of matches that should be generated
    connected_components (list[])

    Returns:
    pandas.DataFrame: Dataframe of the dataset

   """
    '''Returns a list of tuples of a matched pair of products. A pair of products will be considered as matched
    if they are in the same c'''
    global loaded_image_ID
    matches={}; edges=[] 
    connected_components = flatten(connected_components)
    for c in connected_components: edges.extend(list(c.edges))
    while True:
        m1, m2 = edges[np.random.choice(len(edges))]
        if m1 not in loaded_image_ID:            
            try:
                get_image(m1, df)
                loaded_image_ID[m1]=1
            except: print(m1, ' not loaded'); continue
        if m2 not in loaded_image_ID:            
            try:
                get_image(m2, df)
                loaded_image_ID[m2]=1
            except: print(m2, ' not loaded'); continue
        matches[(m1,m2)]=1
        if len(matches)==num: break
    return matches