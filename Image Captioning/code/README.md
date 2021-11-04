# Image Captioning Tutorial

This repository is a tutorial for image captioning using PyTorch. The model contains three architectures: LSTM, RNN, LSTM (feature concatenated).

Implementation Authors: Zirui Wang (Colin), Yung-Chieh Chan (Jerry)

Starter Code: Ajit Kumar, Savyasachi

Special thanks to Yunjey Choi for providing starter code in `vocab.py`

## Usage
* After defining the configuration (say `default.json`) - simply run 
```sh
$ python3 main.py default
```
 * to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/default_experiment` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace

## Dependencies
- numpy
- matplotlib
- PIL
- PyTorch
- COCO
- nltk
