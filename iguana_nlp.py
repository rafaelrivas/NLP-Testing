import pandas as pd
pd.set_option('max_colwidth',100)
pd.set_option('max_rows', 1000)

#reading the csv with the spanish labeled tweets

ls

cd Desktop/dev/ml/nlp


data1 = pd.read_csv('tw_faces4tassTrain1000rc.csv', encoding='utf-8')
data2 = pd.read_csv('stompol-tweets-train-tagged.csv', encoding='utf-8')
data3 = pd.read_csv('socialtv-tweets-train-tagged.csv', encoding='utf-8')
data4 = pd.read_csv('TASS2017_T1_development.csv', encoding='utf-8')
data5 = pd.read_csv('general-tweets-train-tagged-2016.csv', encoding='utf-8')
data6 = pd.read_csv('general1k_csv.csv', encoding='utf-8')
data7 = pd.read_csv('TASS2017_T1.csv', encoding='utf-8')
data8 = pd.read_csv('general2016_csv.csv', encoding='utf-8')

tweets_corpus = pd.concat([
        data1,
        data2,
        data3,
        data4,
        data5,
        data6,
        data7,
        data8
    ])

tweets_corpus.drop(['tweet_id'], axis=1)
tweets_corpus.sample(20)
tweets_corpus.info()
tweets_corpus = tweets_corpus.query('polarity == "P" or polarity == "N" or polarity == "P+"')

#remove links
tweets_corpus = tweets_corpus[-tweets_corpus.content.str.contains('^http.*$')]
tweets_corpus.shape

#download spanish stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation

spanish_stopwords = stopwords.words('spanish')
non_words = list(punctuation)

#we add spanish punctuation
non_words.extend(['¿', '¡'])
non_words.extend(map(str,range(10)))
non_words

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = SnowballStemmer('spanish')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = ''.join([c for c in text if c not in non_words])
    # tokenize
    tokens =  word_tokenize(text)

    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


# Model Evaluation
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

tweets_corpus['polarity_bin'] = 0
tweets_corpus.polarity_bin[tweets_corpus.polarity.isin(['P', 'P+'])] = 1
tweets_corpus.polarity_bin.value_counts(normalize=True)

# Finding optimal hyperparameters
vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokenize,
                lowercase = True,
                stop_words = spanish_stopwords)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('cls', LinearSVC()),
])

parameters = {
    'vect__max_df': (0.5, 1.9),
    'vect__min_df': (10, 20,50),
    'vect__max_features': (500, 1000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'cls__C': (0.2, 0.5, 0.7),
    'cls__loss': ('hinge', 'squared_hinge'),
    'cls__max_iter': (500, 1000)
}


grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1 , scoring='roc_auc')
grid_search.fit(tweets_corpus.content, tweets_corpus.polarity_bin)

#Finding best Param
grid_search.best_params_

#Save the Model
from sklearn.externals import joblib
joblib.dump(grid_search, 'grid_search.pkl')

#We do crossvalidation here to show the performance of the model
model = LinearSVC(C=.2, loss='squared_hinge',max_iter=500,multi_class='ovr',
              random_state=None,
              penalty='l2',
              tol=0.0001
)

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = spanish_stopwords,
    min_df = 50,
    max_df = 1.9,
    ngram_range=(1, 1),
    max_features=1000
)

corpus_data_features = vectorizer.fit_transform(tweets_corpus.content)
corpus_data_features_nd = corpus_data_features.toarray()

scores = cross_val_score(
    model,
    corpus_data_features_nd[0:len(tweets_corpus)],
    y=tweets_corpus.polarity_bin,
    scoring='roc_auc',
    cv=5
    )

scores.mean()

# polarity Prediction
tweets = pd.read_csv('rating_iguana.csv', encoding='utf-8')

# Language detection
tweets.head()

import langid
from langdetect import detect
import textblob

def langid_safe(tweet):
    try:
        return langid.classify(tweet)
    except Exception as e:
        pass

def langdetect_safe(tweet):
    try:
        return detect(tweet)
    except Exception as e:
        pass

def textblob_safe(tweet):
    try:
        return textblob.TextBlob(tweet).detect_language()
    except Exception as e:
        pass

sample_tweets = pd.DataFrame()
sample_tweets['rating'] = tweets.body
sample_tweets['lang_langid'] = sample_tweets.rating.apply(langid_safe)
sample_tweets['lang_langdetect'] = sample_tweets.rating.apply(langdetect_safe)
sample_tweets['lang_textblob'] = sample_tweets.rating.apply(textblob_safe)

sample_tweets

sample_tweets.to_csv('reviews_parsed2.csv', encoding='utf-8')

sample_tweets = sample_tweets.query(''' lang_langdetect == 'es' or lang_langid == 'es' or lang_textblob == 'es'  ''')
sample_tweets.shape

# Building Pipeline
pipeline.fit(tweets_corpus.content, tweets_corpus.polarity_bin)

pipeline.fit(tweets_corpus.content, tweets_corpus.polarity_bin)
sample_tweets['polarity'] = pipeline.predict(sample_tweets.rating)

# Saving the final result into a csv
sample_tweets[['rating', 'polarity']].to_csv('ratings_validados.csv', encoding='utf-8')
