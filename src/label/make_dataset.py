import jsonlines
from sklearn.model_selection import train_test_split


def serialize(df, match_list, mismatch_list):
    """Returns list of tuple (serialized first product, serialized second product, label).
    Matched pairs have label 1, mismatched label 0."""
    dataset = []
    for ID1, ID2 in match_list:
        m1, m2 = df[df.asin == ID1]['prod_info'].values[0], df[df.asin == ID2]['prod_info'].values[0]
        dataset.append((ID1, ID2, m1, m2, 1))
    for ID1, ID2 in mismatch_list:
        m1, m2 = df[df.asin == ID1]['prod_info'].values[0], df[df.asin == ID2]['prod_info'].values[0]
        dataset.append((ID1, ID2, m1, m2, 0))

    return dataset


def train_val_test_split(data):
    """Returns traing, validation and test dataset with ration 3:1:1"""
    train, tmp = train_test_split(data, train_size=0.6)
    val, test = train_test_split(tmp, train_size=0.5)
    return train, val, test


def write_training_data(data_name, data):
    """Saves training dataset into txt and jsonl-files."""
    print('Writing... ')
    train, val, test = train_val_test_split(data)
    with open(data_name + 'train.txt', 'w') as output, jsonlines.open(data_name + 'train.jsonl', 'w') as js:
        for ID1, ID2, m1, m2, label in train:
            output.write(f"{m1}\t{m2}\t{label}\n")
            js.write({"ID1": f'https://www.amazon.com/dp/{ID1}'
                         , "ID2": f'https://www.amazon.com/dp/{ID2}'
                         , "left": m1, "right": m2, 'label': label})
    with open(data_name + 'valid.txt', 'w') as output, jsonlines.open(data_name + 'valid.jsonl', 'w') as js:
        for ID1, ID2, m1, m2, label in val:
            output.write(f"{m1}\t{m2}\t{label}\n")
            js.write({"ID1": f'https://www.amazon.com/dp/{ID1}'
                         , "ID2": f'https://www.amazon.com/dp/{ID2}'
                         , "left": m1, "right": m2, 'label': label})
    with open(data_name + 'test.txt', 'w') as output, jsonlines.open(data_name + 'test.jsonl', 'w') as js:
        for ID1, ID2, m1, m2, label in test:
            output.write(f"{m1}\t{m2}\t{label}\n")
            js.write({"ID1": f'https://www.amazon.com/dp/{ID1}'
                         , "ID2": f'https://www.amazon.com/dp/{ID2}'
                         , "left": m1, "right": m2, 'label': label})
    print('Done!')


def write_triplets_data(data_name, data):
    print('Writing... ')
    train, val, test = train_val_test_split(data)
    with jsonlines.open(data_name + 'train_img_triplets.jsonl', 'w') as js:
        for ID1, ID2, ID3 in train:
            js.write({"anchor": ID1
                         , "positive": ID2
                         , "negative": ID3})
    with jsonlines.open(data_name + 'valid_img_triplets.jsonl', 'w') as js:
        for ID1, ID2, ID3 in val:
            js.write({"anchor": ID1
                         , "positive": ID2
                         , "negative": ID3})
    with jsonlines.open(data_name + 'test_img_triplets.jsonl', 'w') as js:
        for ID1, ID2, ID3 in test:
            js.write({"anchor": ID1
                         , "positive": ID2
                         , "negative": ID3})
    print('Done!')
