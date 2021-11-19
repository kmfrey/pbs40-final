### Combining .h5 data and the information from the genre latex file.
### 10,000 song data (1.8 GB), 133,676 genre labels.
### Total overlap: 1390 songs. 1201 with most popular genres only.
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import copy
from classify_genre import genre_labels, simplified_labels

def collectIds():
    track_ids = []
    with open("C:/Users/kmfre/projects/comp-neuro/data/msd_tagtraum_cd1.cls") as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            fields = line.split('\t')
            track_ids.append(fields[0].strip())
    return track_ids

# If "write_out" is true: write file with absolute paths to track data with associated genre data.
# Output: data_filepaths
def findTracks(write_out = False):
    track_ids = collectIds()
    # can use the name of the file to determine whether it overlaps or not.
    data_filepaths = {}
    count = 0
    for root, dirs, files in os.walk('C:/Users/kmfre/projects/comp-neuro/data/MillionSongSubset'):
        for file in files: 
            for id in track_ids:
                if file.startswith(id):
                    data_filepaths[id] = os.path.join(root, file)
                    track_ids.remove(id) # for slightly quicker iteration

    if write_out:
        with open("C:/Users/kmfre/projects/comp-neuro/data/dataFilepaths.txt", 'w') as output:
            for path in data_filepaths.values():
                output.write(path + '\n')

    return data_filepaths

# Output: File with the absolute paths to track data with only the simplified genre data and associated genre.
def createSimplifiedData():
    id_to_genre = {}
    with open("C:/Users/kmfre/projects/comp-neuro/data/msd_tagtraum_cd1.cls") as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            fields = line.split('\t')
            # if either the first or second genre is in the simplified genre labels
            if fields[1].strip() in simplified_labels:
                id_to_genre[fields[0].strip()] = fields[1].strip()
            elif len(fields) > 2 and fields[2] in simplified_labels:
                id_to_genre[fields[0].strip()] = fields[2].strip()

    filepaths = findTracks()
    id_to_path = copy.copy(filepaths)
    for id in filepaths.keys():
        # remove any path not included in the genre dictionary
        if id not in id_to_genre.keys():
            id_to_path.pop(id)

    with open("C:/Users/kmfre/projects/comp-neuro/data/simplified_data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in id_to_path.keys():
            writer.writerow([item, id_to_genre[item], id_to_path[item]])


# Visualize data
def plot_genres(total_data, title):
    # if total_data = true, look at all data. Otherwise, just the usable songs.
    id_genre = {}
    dict = {}
    with open("C:/Users/kmfre/projects/comp-neuro/data/msd_tagtraum_cd1.cls") as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            fields = line.split('\t')
            id_genre[fields[0].strip()] = fields[1].strip()
    dict = id_genre

    # create the smaller dictionary if needed
    if not(total_data):
        smaller_dataset = get_song_ids()
        smaller_id_genre = {}
        for id in smaller_dataset:
            smaller_id_genre[id] = id_genre[id]
        # assign to dict
        dict = smaller_id_genre


    # calculate number of each type
    num_per_genre = {}
    for genre in genre_labels:
        num_per_genre[genre] = 0
    for g in dict.values():
        num_per_genre[g] += 1
    
    # create an array of the count in the order of genre_labels
    genre_counts = []
    for genre in genre_labels:
        genre_counts.append(num_per_genre[genre])
    # x: genre; y: number of songs
    x = np.arange(len(genre_labels))

    fig, ax = plt.subplots()
    rects = ax.bar(x, genre_counts)
    ax.set_ylabel('Number of songs')
    ax.set_title(title)
    ax.set_xticks(x, genre_labels)
    ax.bar_label(rects, padding=3)

    fig.tight_layout()
    plt.show()


# helper function
def get_song_ids():
    ids = []
    with open("C:/Users/kmfre/projects/comp-neuro/data/dataFilepaths.txt") as file:
        for path in file.readlines():
            idx_start = path.rindex("\\")
            # in form of "TRxxxxxxx.h5", so all but the last 3 characters
            id = path[idx_start+1:-4]
            ids.append(id)
    return ids


# plot_genres(True, "Number of songs per genre")
# plot_genres(False, "Number of songs per genre in dataset.")
# createSimplifiedData()