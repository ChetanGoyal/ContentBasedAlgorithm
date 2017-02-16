# #Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    token = movies.genres
    list_string = []
    for genre_value in token:
        list_string.append(tokenize_string(genre_value))
    movies["tokens"] = list_string
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    list_vocab = set()
    vocab = defaultdict(lambda: 0)
    idf_dict = defaultdict(lambda: 0)
    for token in movies.tokens:
        for item in token:
            list_vocab.add(item)
    for index, value in enumerate(sorted(list_vocab)):
        vocab[value] = index
    for key, value in vocab.items():
        counter_item_document = 0
        for token in movies.tokens:
            for item in token:
                if (key == item):
                    counter_item_document += 1
                    break
        idf_dict[key] = math.log10(len(movies) / counter_item_document)

    row = []
    column = []
    data = []
    for token in movies.tokens:
        count = -1
        tfidf = []
        for key in sorted(idf_dict.keys()):
            count += 1
            freq_of_key = token.count(key)
            if freq_of_key != 0:
                maximum_freq_term = sorted(Counter(token).items(), key=lambda x: -x[1])
                tfidf.append((freq_of_key / maximum_freq_term[0][1]) * idf_dict[key])
            else:
                tfidf.append(0)
            column.append(count)
            row.append(0)
        data.append((csr_matrix((tfidf, (row, column)), dtype=float)))
        row = []
        column = []
    movies["features"] = data
    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    normA = math.sqrt(sum(a.data ** 2))
    normB = math.sqrt(sum(b.data ** 2))
    product = a.dot(b.transpose()).data
    value = (product / (normA * normB))
    return value


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO

    predict_value = []

    for key, value in ratings_test.iterrows():
        userId_test = value['userId']
        movieId_test = value['movieId']
        user_train = ratings_train[ratings_train['userId'] == userId_test]
        movie_testdata = movies[movies['movieId'] == movieId_test]
        movie_test_feature = movie_testdata.iloc[0]['features']
        cosine_sum = 0
        cosine_rating_sum = 0
        rating_sum = 0
        positive_value = 0
        for key1, value1 in user_train.iterrows():
            movie_id = value1['movieId']
            movie_rating = value1['rating']
            movie_traindata = movies[movies['movieId'] == movie_id]
            movie_train_feature = movie_traindata.iloc[0]['features']
            cosine_value = cosine_sim(movie_test_feature, movie_train_feature)
            if cosine_value > 0:
                cosine_sum += cosine_value
                cosine_rating_sum += (cosine_value * movie_rating)
                positive_value += 1
            else:
                rating_sum += movie_rating
        if positive_value > 0:
            predict_value.append(cosine_rating_sum[0] / cosine_sum[0])
        else:
            predict_value.append(rating_sum / len(user_train))
    return np.array(predict_value)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()