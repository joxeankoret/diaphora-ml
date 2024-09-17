# Diaphora ML

This repository contains the [Diaphora](https://github.com/joxeankoret/diaphora) Machine Learning tools and the datasets used by Diaphora to generate the [distributed ML model](https://github.com/joxeankoret/diaphora/blob/master/ml/diaphora-amalgamation-model.pkl).

# What is in this repository?

The directory `tools` contains the following tools:
 * diaphora-export.sh: A bash shell script to export with Diaphora a binary file. Please edit it before to setup your IDA and Diaphora paths.
 * create_dataset.py: Python tool to build a dataset cross comparing binaries that the tool believe are the same. Currently, only the following 2 rules are considered: 1) 99% of the functions have the same address, or 2) 95% of the function names are the same (ignoring auto-generated ones).
 * train_dataset.py: Python tool to train a dataset and build model or check an already trained model against a different dataset. It supports a myriad of other use-cases, please see its help for more details.
 * validate_against_diff.py: Python tool to check a ML model against a saved Diaphora diffing results database. It's mostly used to identify potential false positives with a trained ML model.

The directory `datasets` contains the following files:
 * cisco-talos-dataset-2.csv.bz2: A BZ2 compressed CSV file with the generated dataset cross comparing similar binaries from the Cisco Talos Dataset-2.
 * diaphora-amalgamation.csv.bz2: A BZ2 compressed CSV file with the Diaphora's dataset called "the amalgamation", containing the cross comparison data for the Diaphora's testing suite, issues files, and the Cisco Talos Dataset-2 files.

The directory `example` contains the following example dataset and script:
 * mini-dataset: A directory with a few binary executables. Just an example dataset.
 * run.sh: Example script that exports with Diaphora every single binary in the `mini-dataset` directoy, generates a CSV files cross comparing all the exported binaries, splits the created dataset into the usual training, validation and test datasets; and then trains a model (using a decision tree algorithm) and verifies it against the test and validation datasets. Please remember to change in the file `/tools/diaphora-export.h` the paths for IDA and Diaphora.
 
