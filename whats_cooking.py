import pprint
pr = pprint.PrettyPrinter(indent=4)
import json
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

train_data_file_name = 'Data/train.json'
test_data_file_name = 'Data/test.json'
submission_file_name = 'Submission.csv'

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
            if l.strip() != '},' and l.strip() != '}': 
                s += l
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
def get_data(json_records, do_labels):
    ids = []
    labels = []
    features = []
    for json_rec in json_records:
        ids.append(json_rec['id'])
        if do_labels:
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
    #freq_df = freq.to_frame().iloc[::-1]
    #freq_df.plot.barh()
    print freq

### create, grid search, cross-validate and calculate accuracy score
def train_classifier(clf, params, X, y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    grid_obj = GridSearchCV(clf, params, scoring=make_scorer(accuracy_score))
    grid_obj = grid_obj.fit(X, y)
    
    print 
    print "Training {} classifier - accuracy scores are:".format(clf.__class__.__name__)    
    pr.pprint(grid_obj.grid_scores_)
    
    #return grid_obj.best_estimator_
    

### Pre-processing ###
json_recs = read_json_records(train_data_file_name, count=1000000)
ids_train, ingr_train, y_train = get_data(json_recs, True)

json_recs = read_json_records(test_data_file_name, count=1000000)
ids_test, ingr_test, y_test = get_data(json_recs, False)

vectorizer = CountVectorizer()
ingr_all = ingr_train + ingr_test
vectorizer.fit(ingr_all)
X_train = vectorizer.transform(ingr_train)
X_test= vectorizer.transform(ingr_test)

### Data Exploration ###
explore_labels(y_train)

### Classification exploration ###
#from sklearn.naive_bayes import MultinomialNB
#train_classifier(MultinomialNB(), {'fit_prior': [False]}, X_train, y_train)

from sklearn.svm import SVC
#train_classifier(SVC(), {'C': [0.2], 'kernel': ['linear']}, X_train, y_train)

### training chosen algorithm ###
clf = SVC(C=0.2, kernel='linear')
clf.fit(X_train, y_train, verbose=True)

### making predictions ###
pred = clf.predict(X_test)
explore_labels(pred)

submission = pd.DataFrame({'id': ids_test, 'cuisine': pred})
submission.to_csv(path_or_buf=submission_file_name, columns=['id', 'cuisine'], 
                  index=False)











