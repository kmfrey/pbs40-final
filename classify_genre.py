# Implementing an SVM model to classify songs into genres

import csv

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC

genre_labels = ["Pop_Rock", "Rap", "Jazz", "Vocal", "Electronic", "RnB",
                "Folk", "Blues", "Latin", "Country", "International", "Reggae", "New Age"]

# based on available data, use only labels with > 55 songs
simplified_labels = ["Pop_Rock", "Rap", "Jazz", "Electronic", "Blues", "Country"]
le = preprocessing.LabelEncoder()
le.fit(simplified_labels)

# Getting 48 timbre features from the middle of each song. Retrieve key, time signature, and tempo
# Timbre was chosen partially because https://samyzaf.com/ML/song_year/song_year.html used it for their predictions
# Write into a CSV
def get_features(num_features):
    with open("C:/Users/kmfre/projects/comp-neuro/data/simplified_data.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        output = open("C:/Users/kmfre/projects/comp-neuro/data/genre_features.csv", 'w')
        writer = csv.writer(output, delimiter='\t', quotechar='|')
        # write the header
        header = ['genre', 'key', 'timesignature', 'tempo']
        for i in range(num_features):
            header.append("t"+str(i))
        writer.writerow(header)
        for row in reader:
            path = row[2]
            with h5py.File(path.strip(), 'r') as f:
                timbre = f.get('analysis').get('segments_timbre')
                # get lowest idx
                low_idx = max(int(len(timbre[:, 0])/2) - int(num_features/2), 0)
                max_idx = min(low_idx + num_features, len(timbre[:, 0]))
                # Selecting only 1 of the dimensions (to reduce feature size). Dimension 2, which is shown in the graphs.
                timbre_vector = np.asarray(timbre[low_idx:max_idx, 1])
                # other data (scalar)
                tempo = f.get('analysis').get('songs').fields('tempo')[0]
                key = f.get('analysis').get('songs').fields('key')[0]
                time = f.get('analysis').get('songs').fields('time_signature')[0]
            # write into CSV
            out_row = [row[1], key, time, tempo] + list(map(lambda x: str(x), timbre_vector))
            writer.writerow(out_row)
        output.close()
            

# graph examples of a timbre vector along 1 dimension from 2 songs
def graph_timbre():
    with open("C:/Users/kmfre/projects/comp-neuro/data/dataFilepaths.txt") as paths:
        song1 = paths.readline()
        song2 = paths.readline()
        with h5py.File(song1.strip(), 'r') as f:
            vector = f.get('analysis').get('segments_timbre')[:, 1]
        with h5py.File(song2.strip(), 'r') as f:
            vector2 = f.get('analysis').get('segments_timbre')[:, 1]
    plt.plot(vector, label = "Song 1")
    plt.plot(vector2, label = "Song 2")
    plt.legend()
    plt.title("Timbre of 2 songs along the 2nd dimension.")
    plt.show()

# graph tempo as a histogram
def graph_tempo():
    df = pd.read_csv("C:/Users/kmfre/projects/comp-neuro/data/genre_features.csv", '\t')
    plt.hist(df.tempo, bins=20)
    plt.title("Tempo of data")
    plt.ylabel("Number of songs")
    plt.xlabel("Beats per minute")
    plt.show()

# visualize the data along some dimensions
def visualize_data(X_train):
    # draw scatter for the training data along 2 dimensions (tempo & random timbre)
    plt.scatter(X_train.loc[:, 'tempo'], X_train.loc[:, 't30'])
    plt.title('Data along tempo & t30 dimension')
    plt.xlabel('Tempo')
    plt.ylabel('Timbre')
    plt.show()



def classify():
    # read features into a DF
    df = pd.read_csv("C:/Users/kmfre/projects/comp-neuro/data/genre_features.csv", '\t')
    genres, features_df = preprocess_data(df)
    # split data into training & test. 70% for training, 30% for test.
    X_train, X_test, y_train, y_test = train_test_split(features_df, genres, test_size=0.3)
    # Since the data set is sizeable, we will use a linear kernel.
    classifier = LinearSVC()
    # determine length of time it takes to train
    start = time.time()
    classifier.fit(X_train, y_train)
    print("Time taken to fit model:", time.time()-start)

    y_pred = classifier.predict(X_test)

    # evaluate
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Checking that output is not just pop/rock, which is the most common label.
    # check_pop(y_pred)
    # visualize_data(X_train, classifier)
    return classifier

# function from StackOverflow (https://stackoverflow.com/a/46581125)
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def check_pop(y_pred):
    count = 0
    for y in y_pred:
        if y != le.transform(['Pop_Rock'])[0]:
            count += 1
    print("Number of non-pop/rock predicted songs:", count)

# helper function to clean the dataset & normalize the data.
# Returns a tuple of genres and features
def preprocess_data(dataframe):
    # transform the labels into integers
    df = dataframe
    df.genre = le.transform(df.genre)
    df = clean_dataset(df)
    features_df = df.drop('genre', axis=1)
    # in range of number of features from timbre vector
    timbre_df = features_df.loc[:, 't0':'t47']
    features_df.loc[:, 't0':'t47'] = preprocessing.normalize(timbre_df, axis=0)
    # normalize other features
    features_df.loc[:, 'key':'timesignature'] = preprocessing.normalize(features_df.loc[:, 'key':'timesignature'])
    # robustly scale the tempo, which includes outliers.
    rscaler = preprocessing.RobustScaler(with_centering=False)
    features_df.loc[:, 'tempo'] = rscaler.fit_transform(features_df.loc[:, 'tempo'].values.reshape(-1,1))
    return df.genre, features_df


# get_features(48)
clf = classify()
