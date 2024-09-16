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

import sys
import sqlite3
import argparse

import joblib
import pandas as pd

from common import debug, log, sqlite3_connect
from create_dataset import compare_row, SELECT_FIELDS
from train_dataset import DATA_FRAME_FIELDS

#-------------------------------------------------------------------------------
class CMlDiffValidator:
  def __init__(self):
    self.verbose = False
    self.clf = None
    self.db = None

    self.db_name = None
    self.ml_name = None
    self.cpus = [None, None]

  def __del__(self):
    if self.db is not None:
      self.db.close()

  def connect_database(self, db_name : str):
    self.db = sqlite3_connect(db_name)

    cur = self.db.cursor()
    try:
      sql = "select * from config"
      cur.execute(sql)
      row = cur.fetchone()
      if row is None:
        raise Exception("Invalid Diaphora diffing results database (no `config` table)!")

      log(f"Attaching main database {row[0]}")
      sql = f'attach "{row[0]}" as main_db'
      cur.execute(sql)

      log(f"Attaching diff database {row[1]}")
      sql = f'attach "{row[1]}" as diff_db'
      cur.execute(sql)

      sql = "select processor from main_db.program"
      cur.execute(sql)
      row = cur.fetchone()
      self.cpus[0] = row[0]

      sql = "select processor from diff_db.program"
      cur.execute(sql)
      row = cur.fetchone()
      self.cpus[1] = row[0]
    finally:
      cur.close()

  def load_model(self, ml_name : str):
    self.clf = joblib.load(ml_name)
    log(f"Using model {self.clf}")

  def validate_single_row(self, row : sqlite3.Row):
    d = dict(row)
    ea1 = int(d["address"] , 16)
    ea2 = int(d["address2"], 16)

    sql = f"""select {SELECT_FIELDS}
                from main_db.functions as f,
                     diff_db.functions as df
               where f.address = ?
                 and df.address = ? """
    cur = self.db.cursor()
    try:
      cur.execute(sql, (str(ea1), str(ea2)))
      row = cur.fetchone()
      ret = compare_row(dict(row), False)

      same_arch = self.cpus[0] == self.cpus[1]
      ret.values["cpu"] = same_arch
      ret.values["arch"] = same_arch

      df = pd.DataFrame([ret.values])
      pred = self.clf.predict(df.loc[:,DATA_FRAME_FIELDS])
      prob = self.clf.predict_proba(df.loc[:,DATA_FRAME_FIELDS])

      if pred != 1:
        log("ML differs: match 0x%08x 0x%08x Ratio %f Compare ratio %f ML prediction %d proba %f" % (ea1, ea2, float(d["ratio"]),  ret.values["ratio"], pred, prob[0][1]))
        same_name = d['name'] == d['name2']
        if same_name:
          log(f"ML model false")
    finally:
      cur.close()

  def validate_rows(self):
    cur = self.db.cursor()
    try:
      sql = "select * from results"
      cur.execute(sql)

      while 1:
        row = cur.fetchone()
        if row is None:
          break

        self.validate_single_row(row)

    finally:
      cur.close()

  def validate(self, db_name : str, ml_name : str):
    self.connect_database(db_name)
    self.load_model(ml_name)
    self.validate_rows()

#-------------------------------------------------------------------------------
def validate(args):
  validator = CMlDiffValidator()
  validator.verbose = args.verbose
  validator.validate(args.diffing_database, args.model)

#-------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(prog=sys.argv[0],
                    description='Compare Diaphora diffing results with a trained ML model.')
  parser.add_argument("-d", "--diffing-database", help="Diaphora diffing results database to check.")
  parser.add_argument("-m", "--model", help="File name with the already trained ML model.")
  parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode")
  args = parser.parse_args()
  print(args)

  validate(args)

if __name__ == "__main__":
  main()

