#!/usr/bin/python

"""
Part of Diaphora, a binary diffing tool
Copyright (c) 2015-2024, Joxean Koret

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import time
import sklearn
import argparse
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import neural_network
from sklearn import discriminant_analysis

from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib

from common import debug, log

#-------------------------------------------------------------------------------
GOOD_MATCH_FIELD = "function"
DATA_FRAME_FIELDS = [
  'cpu', 'arch', 'ratio', 'nodes', 'min_nodes', 'max_nodes', 'edges',
  'min_edges', 'max_edges', 'pseudocode_primes', 'strongly_connected',
  'min_strongly_connected', 'max_strongly_connected', 'strongly_connected_spp',
  'loops', 'min_loops', 'max_loops', 'constants', 'source_file'
]

#-------------------------------------------------------------------------------
CLASSIFIER_TYPES = [
  discriminant_analysis.LinearDiscriminantAnalysis,
  discriminant_analysis.QuadraticDiscriminantAnalysis,
  ensemble.AdaBoostClassifier,
  ensemble.BaggingClassifier,
  ensemble.ExtraTreesClassifier,
  ensemble.GradientBoostingClassifier,
  ensemble.RandomForestClassifier,
  linear_model.BayesianRidge,
  linear_model.LogisticRegression,
  linear_model.SGDClassifier,
  naive_bayes.BernoulliNB,
  naive_bayes.GaussianNB,
  naive_bayes.MultinomialNB,
  neighbors.KNeighborsClassifier,
  neural_network.MLPClassifier,
  svm.SVC,
  svm.SVR,
  tree.DecisionTreeClassifier,
]

#-------------------------------------------------------------------------------
class CDiaphoraClassifier:
  def __init__(self, classifier_type = 0):
    self.X = []
    self.y = []
    self.clf = None
    self.criterion = None
    self.classifier_type = CLASSIFIER_TYPES[classifier_type]

    self.verbose = True
    self.do_evaluate_model = True
    self.find_best_params = True
    self.show_confusion_matrix = False

  def log(self, msg):
    if self.verbose:
      log(msg)

  def load_data(self, dataset="dataset.csv"):
    df = pd.read_csv(dataset)
    X = df.loc[:,DATA_FRAME_FIELDS]
    y = df.loc[:,GOOD_MATCH_FIELD]
    return X, y

  def evaluate_model(self, title):
    X = self.X
    y = self.y

    predictions = self.clf.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    self.log(f"Accuracy : {accuracy}")
    self.log(f"Precision: {precision}")
    self.log(f"Recall   : {recall}")
    self.log(f"F1       : {f1}")

    if self.show_confusion_matrix:
      cm = confusion_matrix(y, predictions, labels=self.clf.classes_)
      self.log(f"Confusion matrix:\n{cm}")
      """
      disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
      disp.plot()
      disp.ax_.set_title(title)
      """

  def train(self, dataset, model):
    self.log(f"Loading data from {dataset}...")
    self.X, self.y = self.load_data(dataset)
    log(f"Fitting data with {self.classifier_type.__name__}()...")
    self.clf = self.classifier_type()
    self.clf.verbose = self.verbose

    """
    if self.find_best_params and False:
      from sklearn.model_selection import GridSearchCV

      # {'activation': 'relu', 'hidden_layer_sizes': (60, 30), 'max_iter': 3000, 'solver': 'adam'}

      params = {
          'criterion':  ['gini', 'entropy', 'log_loss'],
          'max_depth':  [None, 6, 8, 10],
          'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
          'splitter': ['best', 'random'],
          'class_weight' : [None, 'balanced']
      }

      params = {
        "hidden_layer_sizes" : [(60, 20), (60, 30), (60, 40)],
        "max_iter" : [3000],
        "activation" : ['identity', 'logistic', 'tanh', 'relu'],
        "solver" : ['adam', 'lbfgs', 'sgd']
      }

      gr_clf = GridSearchCV(
          estimator=self.clf,
          param_grid=params,
          n_jobs=7,
          verbose=2,
      )

      gr_clf.fit(self.X, self.y.ravel())
      log(f"Best params: {gr_clf.best_params_}")

      for key in gr_clf.best_params_:
        setattr(self.clf, key, gr_clf.best_params_[key])
    """
    self.clf.fit(self.X, self.y.ravel())

    self.log("Evaluating the model against training dataset...")
    self.evaluate_model("Confusion matrix for training dataset")

    self.log("Saving model...")
    joblib.dump(self.clf, model)
  
  def cross_validate(self, input_model, dataset):
    if self.clf is None:
      self.log(f"Loading model {input_model}")
      self.clf = joblib.load(input_model)
    
    self.log(f"Loading dataset {dataset}")
    self.X, self.y = self.load_data(dataset)

    log(f"Evaluating the model against dataset {dataset}...")
    self.evaluate_model("Confusion matrix for testing dataset")

  def graphviz(self, input_model):
    if self.clf is None:
      self.log("Loading model...")
      self.clf = joblib.load(input_model)

    dot_data = tree.export_graphviz(self.clf, out_file="tmp.dot", \
                                    filled=True, rounded=True, \
                                    special_characters=True)
    os.system("dot -Tx11 tmp.dot")

#-------------------------------------------------------------------------------
def train(args):
  classifier = CDiaphoraClassifier(int(args.use_classifier))
  classifier.do_evaluate_model = args.do_evaluate_model
  classifier.verbose = args.verbose
  classifier.find_best = args.find_best_params
  classifier.show_confusion_matrix = args.show_confusion_matrix
  classifier.train(args.dataset, args.output)

#-------------------------------------------------------------------------------
def cross_validate(args):
  classifier = CDiaphoraClassifier(int(args.use_classifier))
  classifier.do_evaluate_model = args.do_evaluate_model
  classifier.verbose = args.verbose
  classifier.show_confusion_matrix = args.show_confusion_matrix
  classifier.cross_validate(args.input, args.cross_validate)

#-------------------------------------------------------------------------------
def show_graphviz(args):
  classifier = CDiaphoraClassifier(int(args.use_classifier))
  classifier.do_evaluate_model = args.do_evaluate_model
  classifier.verbose = args.verbose
  classifier.graphviz(args.input)

#-------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(prog=sys.argv[0],
                    description="Generic tool for training and testing datasets with SciKit Learning",
                    epilog="Part of Diaphora, a binary diffing tool.")
  parser.add_argument("-d", "--dataset", default="dataset.csv", help="Dataset to use for training or testing.")
  parser.add_argument("-o", "--output", default="clf.pkl", help="Output file name for the trained model.")
  parser.add_argument("-i", "--input", default="clf.pkl", help="Input file name for an already trained model.")
  parser.add_argument("-t", "--train", action="store_true", help="Train a model.")
  parser.add_argument("-c", "--cross-validate", help="Cross validate the model using the specified dataset.")
  parser.add_argument("-u", "--use-classifier", default=0, choices=list(map(str, range(len(CLASSIFIER_TYPES)))), help="Use the specified classifier.")
  parser.add_argument("-l", "--list-classifiers", action="store_true", help="List the supported classifiers.")
  parser.add_argument("--do-evaluate-model", action="store_true", default=True, help="Evaluate the model and show related data.")
  parser.add_argument("--graphviz", action="store_true", help="Plot a decision tree")
  parser.add_argument("--find-best-params", action="store_true", default=True, help="Find the best params to train the model.")
  parser.add_argument("-m", "--show-confusion-matrix", action="store_true", help="Plot the confusion matrix")
  parser.add_argument("-v", "--verbose", action="store_true", default=False)
  args = parser.parse_args()
  # print(args)

  if args.list_classifiers:
    print("Supported classifiers:\n")
    for index, classifier in enumerate(CLASSIFIER_TYPES):
      print(f"\t{index} -> {classifier.__name__}")
    print("\n")
    return

  t = time.time()
  print_time = False
  if args.train:
    train(args)
    print_time = True

  if args.cross_validate:
    cross_validate(args)
    print_time = True

  if print_time:
    log(f"Done in {time.time() - t} second(s)")

  if args.show_confusion_matrix:
    plt.show()

  if args.graphviz:
    show_graphviz(args)

if __name__ == "__main__":
  main()

