#!/bin/sh

#
# Run this script to generate a dataset out of the binaries in the directory with
# the name 'mini-dataset'. Remember to configure the IDA and Diaphora scripts
# paths in ../tools/diaphora-export.sh
#

# Run a maximum of ${CPUS} processes
CPUS=4

# Find all binaries and export them with Diaphora
find mini-dataset/ -type f | parallel -u -j ${CPUS} ../tools/diaphora-export.sh {}
# Create a CSV dataset cross comparing the generated SQLite databases
../tools/create_dataset.py mini-dataset -o mini-dataset-example.csv
# Split the dataset into training, validation and testing
../tools/split-dataset.py mini-dataset-example.csv
# Then, train a model using adecision tree classifier and output it to mini-dataset.pkl
../tools/train_dataset.py -u 17 -v -o mini-dataset.pkl -t -d mini-dataset-example-train.csv
# And validate it against both the test & training subsets
../tools/train_dataset.py -v -i mini-dataset.pkl -c mini-dataset-example-test.csv
../tools/train_dataset.py -v -i mini-dataset.pkl -c mini-dataset-example-validate.csv

