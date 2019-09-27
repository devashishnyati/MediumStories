

import pandas as pd
import numpy as np



df = pd.read_csv('Data/Medium_AggregatedData.csv', sep=',')



df = df[df['language']=='en']



df_useful_columns = df.drop(['audioVersionDurationSec', 'codeBlockCount','isSubscriptionLocked','publicationdescription','publicationdomain','publicationfacebookPageName','publicationfollowerCount','publicationname','publicationpublicEmail',
                                                              'publicationslug','publicationtags','bio','userId','userName','usersFollowedByCount','usersFollowedCount','scrappedDate','codeBlock', 'publicationtwitterUsername', 'author'], axis=1)



df_drop_duplicate = df_useful_columns.drop_duplicates(subset='postId')



df_drop_duplicate.to_csv('Data/dropduplicate.csv', sep=',')



df_dummy = df_drop_duplicate.head(3)


df_dummy.to_csv('Data/shortdummy.csv',sep=',')



tag = df_useful_columns.groupby('tag_name').size().sort_values(ascending=False)[:10]
top = list(tag.index)
value = list(tag.values)



import matplotlib.pyplot as plt
import seaborn as sns



plt.pie(value, labels=top)
plt.axis('equal')
plt.show()



df[df['postId']=='100139913e4c'].to_csv('Data/same2.csv',sep=',')


from wordcloud import WordCloud
from wordcloud import STOPWORDS


text = ""
for ind, row in df_drop_duplicate.iterrows():
    text += str(row["title"]) + " "
text = text.strip()


plt.figure(figsize=(10,8))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=100, max_words=40).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()




df_test = df_drop_duplicate[df_drop_duplicate['totalClapCount']!=0]




df_new = df_test.groupby('totalClapCount').size()




sns.distplot(df_test['totalClapCount'])




df_drop_duplicate['text']




df_drop_duplicate['text'].to_csv('train.txt', index=False)


import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5




path = 'train.txt'
text = open(path).read().lower()
print('corpus length:', len(text))




text = text[:1000000]




chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f'unique chars: {len(chars)}')




SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples: {len(sentences)}')





X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1






model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))




optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=5, shuffle=True).history




model.save('rnn.h5')
pickle.dump(history, open("history.p", "wb"))




plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');




plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');




def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
        
    return x




prepare_input("This is an example of input for our LSTM".lower())




def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)




def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion




def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]




quotes = ['A major private IT company implements blockchain, artificial intelligence, and Internet of Things t',
         'Thus legalized robots will irrefutably damage the sex trade',
         'Since the launch of the Watson Visual Recognition API',
         'Hi I am Devashish and I am a Data Scientist']




for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()




import matplotlib.pyplot as plt



#1. Regression R2 and RMSE
#2. Classification F1 and Confusion Matrix
#3. Classification Accuracy
#4. Support Vector




# Linear Regression
# Random Forest Regrssion
# Gradient Boosting
r2 = [0.9748, 0.1346, 0.9307, -0.0442]
rmse = [7832.70, 269310.34, 21561.705, 325002.89]
reg = ['LR', 'RFR', 'GB', 'SVR']



plt.bar(reg, r2)
plt.xlabel('Regression Models')
plt.ylabel('R^2 Score')
plt.title('Evaluation Using R^2 Score')



plt.bar(reg, rmse)
plt.xlabel('Regression Models')
plt.ylabel('RMSE Score')
plt.title('Evaluation Using RMSE')



sns.regplot('readingTime', 'totalClapCount', data=df_drop_duplicate, order=3)
plt.show()



print('hi')



max(df_drop_duplicate['totalClapCount'])



df_drop_duplicate.loc[df_drop_duplicate['totalClapCount'].idxmax()]



acc = [0.512, 0.564]
clas = ['GNB', 'MLKNN']



plt.bar(clas, acc)
plt.xlabel('Classification Models')
plt.ylabel('Accuracy Score')
plt.title('Evaluation Using Accuracy')



acc = [0.6670946816197028, 0.7567689992608084]
clas = ['GNB', 'MLKNN']



plt.bar(clas, acc)
plt.xlabel('Classification Models')
plt.ylabel('F1 Score')
plt.title('Evaluation Using F1')

