class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return "Negative"
        elif self.score == 3:
            return "Neutral"
        else:
            return "Positive"


import json

file_name = './data/Books_small.json'

Rev = []
with open(file_name) as f:
    for line in f:
        # print(line)
        r = json.loads(line)
        reviews = Review(r['reviewText'], r['overall'])
        Rev.append(reviews)  # Rev is a list of objects of class Review

print(Rev[5].text)
from sklearn.model_selection import train_test_split
training , test = train_test_split(Rev, test_size=0.33 ,random_state = 42 )
len(training)
print(training[0].text)
train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.fit_transform(test_x)
#print(train_x[0])
print(train_x_vectors[0])

#from sklearn import svm
#clf_svm = svm.SVC(kernel='linear')
#clf_svm.fit(train_x_vectors,train_y)
#test_x[0]
#print(test_x_vectors[0])
from sklearn.svm import SVC
clf_svm = SVC(kernel='linear')
print(clf_svm.fit(train_x_vectors,train_y))
print(clf_svm.predict(train_x_vectors[0])
