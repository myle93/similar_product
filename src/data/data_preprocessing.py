import re
import pandas as pd
from bs4 import BeautifulSoup
import json
import gzip

def parse_dataset(filename):
    """Converts the input file into pandas dataframe. Duplicated line will be removed.

    Parameters:
    filename (str): Input file of type gz

    Returns:
    pandas.DataFrame: Dataframe of the dataset

   """
    data = []
    with gzip.open(filename) as f:
        for i,l in enumerate(f):
            data.append(json.loads(l.strip()))

    # total length of list, this number equals total number of products
    print('Total length of input list: ',len(data))
    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    df3 = df.fillna('')
    # df4 = df3[df3.title.str.contains('getTime')] # unformatted rows
    df5 = df3[~df3.title.str.contains('getTime')] # filter those unformatted rows
    df5 = df5.drop_duplicates(subset='asin')
    print('Number of rows after eliminating duplicated asin: ', len(df5))
    return df5

def get_similar_list(similar_item):
    """In original files the similar_item column contains html strings that contains a list of 
    ASIN of similar products. This method converts these html strings into lists of ASIN.

    Parameters:
    similar_item (str): Input html string
    
    Returns:
    list[str]: List of ASINs of similar products

   """
    soup = BeautifulSoup(similar_item, 'html.parser')
    tmp = [i['href'] for i in soup.find_all('a')]
    res = []
    for t in tmp:
        match = re.search(r'^/dp/(.+)/',t)
        if match: res.append(match.group(1))
    return res

def parse_column(df):
    """Parsed data in relevant columns.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Dataframe of filtered products

   """
    print('Original total number of rows: ', len(df))
    df['feature'] = df['feature'].apply(lambda x: '' if len(x) == 0 else x)
    df['description'] = df['description'].apply(lambda x: '' if len(x) == 0 else x)
    df['category'] = df['category'].apply(lambda x: '' if len(x) == 0 else x)
    df['similar_item'] = df['similar_item'].apply(lambda x: '' if len(x) == 0 else get_similar_list(x))
    df.reset_index(drop=True, inplace=True)
    print('Number of rows after filtering: ', len(df))    
    return df