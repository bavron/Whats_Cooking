import pprint
pr = pprint.PrettyPrinter(indent=4)
import json
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns

train_data_file_name = 'Data/train.json'

### look at the first lines in the file ###
def check_json_file(fn):
    with open(fn, 'r') as f:
        line_count = 0
        while f.readline():
            line_count += 1
    return line_count
        
### reading json records one-by-one ###
def read_json_records(fn, count=None):
    print "Reading records from file '{}' (up to {} records)...".format(fn, count)    
    line_count = check_json_file(fn)
    result = []
    with open(fn, 'r') as f:
        f.readline()
        i, x = (0,0)
        s = ''
        while x < line_count:
            x += 1
            l = f.readline()
            if l.strip() != '},': s += l
            else:
                s += "}"
                result.append(json.loads(s))
                s = ''
                i += 1
                if count != None and i >= count: break
    print "{} records read.".format(len(result))
    return result

### convert the json records to a list of record ids, lables and features ###
### as one string                                                         ###
def get_data(json_records):
    ids = []
    labels = []
    features = []
    for json_rec in json_records:
        ids.append(json_rec['id'])
        labels.append(json_rec['cuisine'])
        ingr_list = json_rec['ingredients']
        ingr = ''
        for i in ingr_list:
            token = ''
            for x in i.split():
                token += x + "_"
            ingr += token[:-1] + " "
        features.append(ingr[:-1])
    return (ids, features, labels)

### encode the ingredients as numbers ###
def vectorize_ingredients(ingrs):
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(ingrs)
    return vectorizer.get_feature_names(), X

### explore labels ###
def explore_labels(y):
    freq = pd.Series(y).value_counts()[:10]
    freq = freq.apply(lambda x: x / float(len(y)))
    freq_df = freq.to_frame().iloc[::-1]
    freq_df.plot.barh()
    print freq

### Pre-processing ###
json_recs = read_json_records(train_data_file_name, count=200)
ids_train, ingr_train, y_train = get_data(json_recs)
ingr_names, X_train = vectorize_ingredients(ingr_train)

### Data Exploration ###
explore_labels(y_train)


























