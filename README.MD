*Author* : JK Kang
*Contact* : jk3587@gmail.com

# Cheering/Applause Detector

This project aims to create an end-to-end pipeline where a user would input a video (YouTube link). The video's audio will get analyzed for any cheering or applauses that occurs in the video. The output would be a csv that lists every occurrence of cheering/applause's time stamp as well as the subtitle at that time.

### General Use Case Workflow
1. User inputs a YouTube link (`run_on_own_audiofile.ipynb`)
    - Results .csv
2. Explore subtitles and the results csv from previous step (`Explore_Subtitles_vs_Predictions.ipynb`)

### Potential Business Value
This workflow can be refined and applied to quantify reception metrics for sporting matches, comedy shows, popular speeches, etc. It can be used as vehicle to answer an audience's taste in humor-- a rather esoteric subject.

## Table of Contents
TO_DO Later

## Setup and Requirements
Requirements are in a `environment.yaml` file.

If you have Anaconda installed, run the following line:
``` shell
conda env create -f environment.yaml
````

### Minimum Computing Requirements
This is the minimum computing requirements to run the model.

## Retrieving Timestamp and Predictions

In the iPython notebook, there are several URLs that points to popular speeches such as "I Have a Dream" and the State of the Union from 2019.
By calling the `download audio` function, you can put in any url, and itould download the audio with the subtitles.

The clipsize is how long of a segment the model looks at whether there is cheering or applause. The default is `1000` (aka 1000ms). 

A prediction amount is assigned to the segment using the model located in `Models/v2_cheer_applause_LSTM_ThreeLayer_100Epochs.h5`.

Side Note: Please refer to the "Training Your Own Model" in the Appendix if you would want to use a different model. 

The `save_embeddings_hdf` function from the `utils.py` module, will save the list of embeddings for audio files.

Combine the resulting predictions (list format) with a timespace (linspace derived from clipsize) to create a dataframe which can be then saved as a csv.

Here is a sample table:

![IMAGE](resources/C6A1A6A7CFDB5A98ADAA7ECFCF813573.jpg =210x299)

## Exploring Subtitles vs. Predictions

The second part of the workflow is to analyze what was presumably said during the time when there was a cheer/applause.
This was done by extracting the subtitles and aligning the timestamps together.

TO_DO

## Appendix

### About the Data
The dataset is called Audioset from https://research.google.com/audioset/.
It is a large-scale dataset of manually annotated audio events.

It consists of 632 audio event classes and a collection of 2M+ human-labeled 10-second sound clips from YouTube videos.It covers a wide range of human and animal sounds, musical instruments and sgenres, and common everyday environmental sounds.

For this particular data pipeline, I selected to train on the "cheers" and "applause" sound.

### About the VGGish Embedder
TO_DO write more information regarding the VGGish Embedder


### Various Models (Traditional ML (Logistic Regression, SVM, Tree Classifiers) vs. LSTM)
TO_DO write the different approaches used

### Training Your Own Model
It is possible to train your own model and modify the workflow to finetune what the neural network is looking for.

Currently, this workflow uses the models located in this directory `Models/v2_cheer_applause_LSTM_ThreeLayer_100Epochs.h5.


### TO DO LIST
- augment positive label data (label 66, 67 in Audioset)
- retrain model with augmented data
- run inference with longer clipsize (~2000ms)
- group predicted timestamps to find start and end of every instance of cheer/applause (try non-max suppression?)
- analyze/plot predicted labels
- gather sentences before each predicted timestamp
- NLP on the gathered sentences (find topics, sentiment, etc)
- Analyze videos of speakers over time and find common topics

