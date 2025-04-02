import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
msg=pd.read_csv('naivetext.csv',names=['message','label'])
print("The dimention of the dataset" ,msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
print(X)
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print("\nThe total no of training data",y_train.shape)
print("\nThe total no of testing data",y_test.shape)
vectorizer=CountVectorizer()
X_train_dtm=vectorizer.fit_transform(X_train)
X_test_dtm=vectorizer.transform(X_test)
print("The words or token in test document\n")
print(vectorizer.get_feature_names())
clf=MultinomialNB()
clf.fit(X_train_dtm,y_train)
y_pred=clf.predict(X_test_dtm)
print("\n Accuracy of classifier ",metrics.accuracy_score(y_test,y_pred))
print("\Confusion matrix")
print(metrics.confusion_matrix(y_test,y_pred))
print("\n The value of precision ",metrics.precision_score(y_test,y_pred))
print("\n The value of recall",metrics.recall_score(y_test,y_pred))
