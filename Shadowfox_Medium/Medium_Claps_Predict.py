


# # Goal 1 - What sort of stories become prominent on Medium?
# We will perform a multivariate regression of views onto multiple independent variables like word count, reading time, total clap count, and others.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import string
import re





df = pd.read_csv("Medium_AggregatedData.csv")






df = df[df['totalClapCount'] != 0]





df.drop(columns=['audioVersionDurationSec', 'codeBlock', 'codeBlockCount', 'collectionId', 'createdDate', 
                 'createdDatetime', 'firstPublishedDate','firstPublishedDatetime','imageCount', 'isSubscriptionLocked',
                 'latestPublishedDate', 'latestPublishedDatetime', 'linksCount', 'responsesCreatedCount', 
                 'socialRecommendsCount','uniqueSlug', 'updatedDate', 'updatedDatetime', 'url', 'vote',
                 'publicationdomain','publicationfacebookPageName', 'publicationdescription',
                 'publicationfollowerCount','publicationname', 'publicationpublicEmail', 'publicationslug',
                 'publicationtags', 'publicationtwitterUsername','slug','name', 'postCount',
                 'usersFollowedByCount', 'usersFollowedCount', 'scrappedDate'], inplace=True)




df.isnull().any()




df = df.dropna(axis=0, subset=['subTitle', 'title', 'bio'])






df = df[:5000]





df['title_len'] = df['title'].str.len()
df['text_len'] = df['text'].str.len()





df['title'] = df['title'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: x.lower())
df['author'] = df['author'].apply(lambda x: x.lower())





df['title_clean'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))





df['text_clean'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))





df['title_clean'] = df['title'].apply(lambda x: re.sub('[' + string.punctuation + '—]', '', x))
df['text_clean'] = df['text'].apply(lambda x: re.sub('[' + string.punctuation + '—]', '', x))

df['title_clean'] = df['title_clean'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))
df['text_clean'] = df['text_clean'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))

df['title_clean'] = df['title_clean'].apply(lambda x: re.sub(' +', ' ', x))
df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(' +', ' ', x))

df['title_clean_len'] = df['title_clean'].str.len()
df['text_clean_len'] = df['text_clean'].str.len()

df['full_text'] = df['author'] + ' ' + df['title_clean'] + ' ' + df['text_clean']





vectorizer = TfidfVectorizer(max_features=None)
full_text_features = vectorizer.fit_transform(df['full_text'])





scaler = StandardScaler()
num_features = scaler.fit_transform(df[['readingTime', 'title_len', 'text_len', 'title_clean_len', 'text_clean_len']])





full_text_features = np.concatenate([full_text_features.toarray(), num_features], axis=1)





X_train, X_test, y_train, y_test = train_test_split(full_text_features, df[['totalClapCount']].values, test_size=0.3)


# # Linear Regression




reg = LinearRegression().fit(X_train, y_train)





y_pred = reg.predict(X_test)






r2_score(y_test, y_pred)





from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)





actual_values = y_test
plt.scatter(y_pred, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Claps')
plt.ylabel('Actual Claps')
plt.title('Linear Regression Model')


# # Random Forest Regressor




from sklearn import ensemble
lr2 = ensemble.RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,
                                     random_state =20, max_features = "sqrt", min_samples_leaf = 30)





modelRF = lr2.fit(X_train, y_train)





print ("R^2 is: \n", modelRF.score(X_test, y_test))





RFRpredictions = modelRF.predict(X_test)





print ('RMSE is: \n', mean_squared_error(y_test, RFRpredictions))





actual_values = y_test
plt.scatter(RFRpredictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Claps')
plt.ylabel('Actual Claps')
plt.title('Random Forest Regression Model')


# # Gradient Boosting Regressor




lr = ensemble.GradientBoostingRegressor()





modelGR = lr.fit(X_train, y_train)





print ("R^2 is: \n", modelGR.score(X_test, y_test))





GBRpredictions = modelGR.predict(X_test)





print ('RMSE is: \n', mean_squared_error(y_test, GBRpredictions))





actual_values = y_test
plt.scatter(GBRpredictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Claps')
plt.ylabel('Actual Claps')
plt.title('Gradient Boosting Regression Model')


# # Support Vector Regression




from sklearn.svm import SVR





clf = SVR()





modelsvr = clf.fit(X_train, y_train)





print ("R^2 is: \n", modelsvr.score(X_test, y_test))





svrpredictions = modelsvr.predict(X_test)





actual_values = y_test
plt.scatter(svrpredictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Claps')
plt.ylabel('Actual Claps')
plt.title('SVR Regression Model')





print ('RMSE is: \n', mean_squared_error(y_test, svrpredictions))






