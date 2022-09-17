from collections import Counter
import warnings

warnings.simplefilter(action='ignore')
import networkx as nx
import copy


def make_graph(df):
    """Returns graph of similar item in dataframe df"""
    H = nx.Graph()
    for i in range(len(df)):
        ID = df.loc[i].asin
        sim = df.loc[i].similar_item
        H.add_node(ID)
        H.add_nodes_from(sim)
        for s in sim:
            H.add_edge(ID, s)
    return H


def make_connected_component(df):
    """Returns list of connected components of similar item in dataframe"""
    H = make_graph(df)
    Hcc = sorted(nx.connected_components(H), key=len, reverse=True)
    res = [H.subgraph(Hcc[i]) for i in range(len(Hcc))]
    res = [i for i in res if len(i.nodes) > 15]
    return res


def get_IDList(df):
    """Returns a list of product ID of input dataframe"""
    ID_list = {}
    for ID in df.asin:
        ID_list[ID] = 1
    return ID_list


def parse_price(string):
    """Format the price column in dataset"""
    res = ''
    try:
        res = float(string[1:])
    except:
        pass
    return res


def clean_dataframe(df, att_list):
    """Removes rows of dataframe with missing info (title, feature, description, similar_item, category).
    Make new column "prod_info" that of serialized textual product information."""
    print('Original total number of rows: ', len(df))
    for att in att_list:
        if att == "parsed_price":
            df = df[df.price != '']
            df['parsed_price'] = df.price.apply(lambda x: parse_price(x))
            df = df[df.parsed_price != '']
        elif att == "image":
            df['image'] = df.image.apply(lambda x: x[0] if len(x) > 0 else '')
            df = df[df.image != '']
        else:
            df = df[df[att] != '']
    df['title'] = df['title'].apply(lambda x: x.replace('\n', ' ').replace('\t', ' '))
    df['feature'] = df['feature'].apply(lambda x: ' '.join(x).replace('\n', ' ').replace('\t', ' '))
    df['description'] = df['description'].apply(lambda x: ' '.join(x).replace('\n', ' ').replace('\t', ' '))
    df['prod_info'] = "COL title VAL \"" + df['title'].map(str) + "\" COL feature VAL \"" + df['feature'].map(str) + \
                      "\" COL description VAL \"" + df['description'].map(str) + "\""
    df.reset_index(drop=True, inplace=True)
    print('Number of rows after filtering: ', len(df))
    return df


def df_component(df, df_cleaned):
    """Returns list of connected components in each subcategory of given dataframe.
    Products with missing information in each component will be then removed.
    """
    df1 = copy.deepcopy(df)
    IDs = get_IDList(df_cleaned)

    def clean_component(connected_components):
        """Removes all nodes in connected components that are not in cleaned dataframe"""
        cleaned_connected_components = []
        for c in connected_components:
            c = nx.Graph(c)
            nodes = list(c.nodes)
            for node in nodes:
                if node not in IDs: c.remove_node(node)
            if len(c.nodes) > 10 and len(c.edges) > 0: cleaned_connected_components.append(c)
        return cleaned_connected_components

    df1['category'] = df1['category'].apply(lambda x: '' if len(x) == 0 else ';'.join(x[:3]))
    cat = Counter(df1.category.values).keys()
    print('Number of subcategories: ', len(cat))
    res = []
    for sub_cat in cat:
        if sub_cat == '': continue
        sub_df = df1[df1.category == sub_cat]
        sub_df.reset_index(drop=True, inplace=True)
        connected_components = make_connected_component(sub_df)
        connected_components = clean_component(connected_components)
        if len(connected_components) > 0: res.append(connected_components)
    del df1
    print('Number of connected components: ', len(res))
    return res


def flatten(list1):
    list2 = []

    def helper(list1, list2):
        for element in list1:
            if type(element) != list:
                list2.append(element)
            else:
                helper(element, list2)

    helper(list1, list2)
    return list2
