# ICLabel Dataset

----
## What is ICLabel?
ICLabel is a project aimed at advancing automated electroenephalographic (EEG) independent component (IC) classification. It is comprised of three interlinked parts: the [ICLabel classifier](https://github.com/lucapton/ICLabel), the [ICLabel website](https://iclabel.ucsd.edu/tutorial), and this dataset. The website crowdsources labels for the dataset that in turn is used to train the classifier.

See the accompanying publication [Coming Soon].


----
## What is this dataset?
The ICLabel dataset contains an unlabeled training dataset, several collections of labels for small subset of the training dataset, and a test dataset 130 ICs where each IC was labeled by 6 experts. In total it is comprised of features from hundreds of thousands of unique EEG ICs (millions if you count similar ICs from different processing stages of the same datasets). Roughly 8000 of those have labels, though the actual usable number is typically closer to 6000 depending on which features are being used. The features included are:
* Scalp topography images (32x32 pixel flattened to 740 elements after removing white-space)
* Power spectral densities (1-100 Hz)
* Autocorrelation functions (1 second)
* Equivalent current dipole fits (1 and 2 dipole)
* Hand crafted features (some new and some from previously published classifiers)

The original time series data are not available. All that is provided is included in this repository as is. I realize having the original time series would make this dataset much more versatile, but unfortunately that's not possible.

----
## Usage
1. Load the class, passing any desired options.
2. Load the dataset.

Example:

    icl = ICLabelDataset()
    icl.download_trainset_features()
    icldata = icl.load_semi_supervised()
