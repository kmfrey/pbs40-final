# Programming Final: Zakaiah Frey '22
## PBS 40: Computational Neuroscience
### '21F

## Procedure
1. Download sample of MillionSongSubset (10K songs)
2. Download Tagtraum CD1 dataset (ID: genre1: genre2)
3. Create a dataset of the overlapping data points between the sample & Tagtraum genre labels
4. Using this CSV, extract relevant data from the H5 files. Create a dataset mapping genre to these features.
5. Create LinearSVM model. 

## Creating the dataset
Using code in `generate_data.py`, I created a dataset of 1201 songIds and their associated genres and HDF5 filepaths. This data includes the songs for which I have files with information in the `msd_tagtraum_cd1.cls`, and the genre is one of the top 6. Then, I opened the data with `classify_genre::get_features` and created a CSV including all the features that I want to analyze. This CSV is then read into a pandas DataFrame during model creation.

### Analyzing data
After creating a usable data set, I worked on extracting features. Data extraction took quite a while, as the files were very confusing and contain a multitude of information. The 10K songs include 1.8GB of data, which I greatly reduced. The timbre comes as a *l* x 12 matrix where *l* is some length.

Originally, I did not normalize the data. This also likely caused the model to be slow.

**genre: key, time signature, tempo, timbre_vector (48x1)**

## Model
When first training the model using `sklearn.svm.SVC`, with a linear kernel and a 96-feature timbre vector, the model took 77 minutes to train. The accuracy here was 0.5125. I had not yet normalized the data.

Next, I attempted to use `sklearn.svm.linearSVC` with normalized data, which is shown to scale much better with large datasets. Seeing as the timecourse was something I was worried about originally, the time of the vanilla SVC model is not unexpected but is still unacceptable for training on only 1201 data points.

Lo and behold, by changing to the linearSVC and normalizing data, the time to train the model decreased to 0.031 seconds and the accuracy increased to 0.6791. This also included a decrease from 96 timbre features to 48 features. All settings are the defaults of `sklearn.svm.LinearSVC`.

### Results
I was slightly concerned that the only predictions would be Pop/Rock because it was so heavily overrepresented in the dataset. However, there were 3 predictions (out of 360) that weren't. While those aren't fantastic numbers, it is also not an unexpected result.

## Figures
### Genres
![total_genres_tagtraum](/graphs/genre-song-totalset.png)
*Total number of songs per genre in the Tagtraum dataset. Only referencing primary genre.*

![small_genres_tagtraum](/graphs/genre-song-smallset.png)
*Number of songs per genre in the Tagtraum dataset that have files in the MillionSongSubset. Only the primary genre.*

From this information, I removed any genre that had less than 55 songs with the genre (as primary). This left us with the labels `["Pop_Rock", "Rap", "Jazz", "Electronic", "Blues", "Country"]`.

### Features
![tempo_of_data](/graphs/tempo_of_data.png)
*This is a histogram of the tempo of all 1201 songs. There is a large variety, with an expected center around 100-150 BPM*.

![timbre_example](/graphs/timbre_2D.png)
*The 1st and 2nd dimensions of timbre are graphed together for the first song in the `genre_features.csv` dataset.*

I graphed the timbre for all dimensions of 2 different songs (one rap, one pop/rock). The 2nd dimension was complex and seemed a good choice, as I needed to take a 1D array sample of the timbre of each song.

![timbre_song1_song2](/graphs/timbre.png)
*The timbre along the 2nd dimension of 2 songs, the first and second songs in the `genre_features.csv` dataset.*

The length is different for the two as the sample lengths are different for each song. However, this graph seems to indicate a good difference in each song. I chose to sample from the middle of each vector because of concerns about fade-in/fade-out.

## Resources
I referenced a few StackOverflow questions which were mentioned in comments. Additionally, I used a lot of the documentation from scikit-learn and h5py.
- Field list for MillionSongDataset: http://millionsongdataset.com/pages/field-list/
- Data choice: https://samyzaf.com/ML/song_year/song_year.html
- Debugging an issue where some parts of the dataset were `inf` or `NaN`: https://stackoverflow.com/a/46581125
- SVM tutorial: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
- Increasing speed of SVM: https://stackoverflow.com/a/65543632
- Normalizing data: https://www.journaldev.com/45109/normalize-data-in-python

