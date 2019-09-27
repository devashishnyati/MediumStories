



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import unicodedata
import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer



df = pd.read_csv('Data/Medium_AggregatedData.csv', sep=',')




df = df[df['language']=='en']



df_useful_columns = df.drop(['audioVersionDurationSec', 'codeBlockCount','isSubscriptionLocked','publicationdescription','publicationdomain','publicationfacebookPageName','publicationfollowerCount','publicationname','publicationpublicEmail',
                                                              'publicationslug','publicationtags','bio','userId','userName','usersFollowedByCount','usersFollowedCount','scrappedDate','codeBlock', 'publicationtwitterUsername', 'author'], axis=1)







df_test = df_useful_columns.drop(['collectionId', 'createdDate', 'createdDatetime', 'firstPublishedDate',
       'firstPublishedDatetime', 'imageCount', 'language',
       'latestPublishedDate', 'latestPublishedDatetime', 'linksCount',
       'readingTime', 'recommends', 'responsesCreatedCount',
       'socialRecommendsCount', 'subTitle', 'tagsCount',
       'totalClapCount', 'uniqueSlug', 'updatedDate', 'updatedDatetime', 'url',
       'vote', 'wordCount', 'slug', 'name', 'postCount'], axis=1)






df_dummy = df_test.copy()






tag = df_useful_columns.groupby('tag_name').size().sort_values(ascending=False)[:10]
top = list(tag.index)
value = list(tag.values)




plt.pie(value, labels=top)
plt.axis('equal')
plt.show()




tag_df = pd.DataFrame(tag)







df = pd.merge(tag_df, df_dummy, on=['tag_name'], how='left')





group_df = df.groupby('postId')['tag_name'].apply(list).to_frame()




df_drop_duplicate = df.drop_duplicates(subset='postId')





result = pd.merge(df_drop_duplicate, group_df, on='postId')




result = result.drop(['tag_name_x'], axis=1)




result.set_index('postId', inplace=True)




df1= result[:1000]






s = result['tag_name_y']




newDF= pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)





stopWordList=stopwords.words('english')
stopWordList.remove('no')
stopWordList.remove('not')

def removeAscendingChar(data):
    data=unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return data

def removeCharDigit(text):
    str='`1234567890-=~@#$%^&*()_+[!{;":\'><.,/?"}]'
    for w in text:
        if w in str:
            text=text.replace(w,'')
    return text

lemma=WordNetLemmatizer()
token=ToktokTokenizer()

def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        #print(x)
        listLemma.append(x)
    return text

def stopWordsRemove(text):
    
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    
    removedList=[x for x in wordList if not x in stopWordList]
    text=' '.join(removedList)
    #print(text)
    return text

def PreProcessing(text):
    #text=removeTags(text)
    #print(text)
    text=removeCharDigit(text)
    #print(text)
    text=removeAscendingChar(text)
    #print(text)
    text=lemitizeWords(text)
    #print(text)
    text=stopWordsRemove(text)
    #print(text)
    return(text)




import nltk
nltk.download('wordnet')




totalText=''
for x in df1['text']:
    ps=PreProcessing(x)
    totalText=totalText+" "+ps





from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc=WordCloud(max_font_size=60).generate(totalText)
plt.figure(figsize=(16,12))
plt.imshow(wc, interpolation="bilinear")




import nltk
freqdist = nltk.FreqDist(token.tokenize(totalText))
freqdist
plt.figure(figsize=(16,5))
freqdist.plot(20)




totalText_tit=''
for x in df1['title']:
    ps=PreProcessing(x)
    totalText_tit=totalText_tit+" "+ps




from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc=WordCloud(max_font_size=60).generate(totalText_tit)
plt.figure(figsize=(16,12))
plt.imshow(wc, interpolation="bilinear")




import nltk
freqdist = nltk.FreqDist(token.tokenize(totalText_tit))
freqdist
plt.figure(figsize=(16,5))
freqdist.plot(20)




df_old = pd.merge(df1, newDF, on='postId')





df = df_old.drop([0], axis=1)




df = df_old.drop(['tag_name_y',0], axis=1)




x=df.iloc[:,0:2].values





y=df.iloc[:,2:].values






x=df.iloc[:,0:2].values
y=df.iloc[:,2:-1].values
# using binary relevance
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
x1=df.title
x2=df.text
from pandas import DataFrame
cv=CountVectorizer().fit(x1)
header=DataFrame(cv.transform(x1).todense(),columns=cv.get_feature_names())
cvArticle=CountVectorizer().fit(x2)
article=DataFrame(cvArticle.transform(x2).todense(),columns=cvArticle.get_feature_names())
import pandas as pd
x=pd.concat([header,article],axis=1)







from sklearn.feature_extraction.text import TfidfTransformer
tfidfhead=TfidfTransformer().fit(header)
head=DataFrame(tfidfhead.transform(header).todense())
tfidfart=TfidfTransformer().fit(article)
art=DataFrame(tfidfart.transform(article).todense())
import pandas as pd
x=pd.concat([head,art],axis=1)




from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))

predictions = classifier.predict(xtest.astype(float))
predictions.toarray()
from sklearn.metrics import accuracy_score
accuracy_score(ytest.astype(float),predictions)




# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))

# predict
predictions = classifier.predict(xtest.astype(float))

accuracy_score(ytest.astype(float),predictions)




# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(xtrain.astype(float), ytrain.astype(float))
# predict
predictions_gnb = classifier.predict(xtest.astype(float))

print(accuracy_score(ytest.astype(float),predictions_gnb))




from skmultilearn.adapt import MLkNN

classifier_knn = MLkNN(k=7)

# train
classifier_knn.fit(xtrain.astype(float), ytrain.astype(float))
# predict
predictions_knn = classifier_knn.predict(xtest.astype(float))

print(accuracy_score(ytest.astype(float),predictions_knn))






from sklearn.metrics import f1_score





f1_gnb = f1_score(ytest.astype(float),predictions_gnb.todense(),average='weighted')





print(f1_gnb)





f1_knn = f1_score(ytest.astype(float),predictions_knn.todense(),average='weighted')





print(f1_knn)

