# dota-2-ml-accuracy

A graduate ML Deep Learning Project on win prediction in Dota 2.

By Nestor J. Maysonet and Nikolas Barran.

## Project Outline

In a paper by Semenov et al. draft win prediction classifiers using traditional machine learning methods
achieved accuracies from ~58-73%. Our goal was to take over 50,000 recorded Dota matches and select only
the hero picks for each team as the primary features. After encoding these, we ran our own logistic regression
classifier as a control and then develped a 3-layer linear neural network to train on the processed dataset.

## Results

Our classifier resulted in an accuracy rate of 86.73% which is a marked improvement over experiments conducted
with traditional algorithms.

We used TensorBoard to visualize our accuracy over epochs:

![](/images/acc.png)

And our loss:

![](/images/loss.png)

As we only ran for 100 epochs (we had no CUDA GPUs) we believe that an accuracy of ~90% could be achieved with the same network and most likely
an even higher accuracy with further refinement of the neural network which demonstrates the value that deep learning offers
esports and game companies.

## Instructions to run on your own

Clone the repo `git clone https://github.com/njmaysonet/dota-2-ml-accuracy.git`

You must download the dataset from https://www.kaggle.com/devinanzelmo/dota-2-matches

You **only** need hero_names.csv, matches.csv, and players.csv, this will save you ~500 mb.

Run data_logreg it will output the underlying dataframes, train the logistic regressor +
show the regression stats, and then run neural.py to train the neural network

The data preprocessing does take a bit to run.
