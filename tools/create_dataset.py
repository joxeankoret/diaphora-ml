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

import re
import os
import sys
import time
import json
import sqlite3
import argparse
import threading

from collections import OrderedDict
from multiprocessing import cpu_count

from common import debug, log, sqlite3_connect, int_compare_ratio, quick_ratio

from typing import Iterable, Any

#-------------------------------------------------------------------------------
SAME_BINARY_PERCENT = 99.0
SAME_NAMES_PERCENT = 99.0
INVALID_SCORE = -1
INVALID_VALUE = -2

CSV_SEPARATOR_FIELD = ","

RE_EXPR = "([a-z0-9]+)\-(32|64)\_([a-z0-9]+)\-([0-9\.]+)\-O([0-3]+)\_(.*)\.sqlite"

FIELDS = ["nodes", "edges", "indegree", "outdegree", "cc",
  "primes_value", "clean_pseudo", "pseudocode_primes", "strongly_connected",
  "strongly_connected_spp", "loops", "constants", "source_file"
]

NUM_FIELDS = ["nodes", "edges", "indegree", "outdegree", "cc",
  "strongly_connected", "loops"
]

PRINT_FIELDS = ['ratio', 'nodes', 'min_nodes', 'max_nodes', 'edges', 'min_edges',
  'max_edges', 'indegree', 'min_indegree', 'max_indegree', 'outdegree',
  'min_outdegree', 'max_outdegree', 'cc', 'min_cc', 'max_cc', 'primes_value',
  'clean_pseudo', 'pseudocode_primes', 'strongly_connected',
  'min_strongly_connected', 'max_strongly_connected', 'strongly_connected_spp',
  'loops', 'min_loops', 'max_loops', 'constants', 'source_file',
]

SELECT_FIELDS = """f.name name1,
       f.nodes nodes1,
       f.edges edges1,
       f.indegree indegree1,
       f.outdegree outdegree1,
       f.cyclomatic_complexity cc1,
       f.primes_value primes_value1,
       f.clean_pseudo clean_pseudo1,
       f.pseudocode_primes pseudocode_primes1,
       f.strongly_connected strongly_connected1,
       f.strongly_connected_spp strongly_connected_spp1,
       f.loops loops1,
       f.constants constants1,
       f.source_file source_file1,
       df.name name2,
       df.nodes nodes2,
       df.edges edges2,
       df.indegree indegree2,
       df.outdegree outdegree2,
       df.cyclomatic_complexity cc2,
       df.primes_value primes_value2,
       df.clean_pseudo clean_pseudo2,
       df.pseudocode_primes pseudocode_primes2,
       df.strongly_connected strongly_connected2,
       df.strongly_connected_spp strongly_connected_spp2,
       df.loops loops2,
       df.constants constants2,
       df.source_file source_file2,
       f.id id1,
       df.id id2,
       f.address ea1,
       df.address ea2 """

COMPARE_SAMPLES_SQL = f"""
select {SELECT_FIELDS}
  from main.functions f,
       diff.functions df
 where f.address = df.address
"""

DIFFERENT_SAMPLES_SQL = f"""
select {SELECT_FIELDS}
  from main.functions f,
       diff.functions df
 where f.id != df.id
   and f.id = 10
   and df.id in (
	select t1.id
	  from main.functions as t1
	  join (
	select id
	  from diff.functions
	 where id != ?
	 order by random()
	 limit ?) as t2
		on t1.id=t2.id
    )
"""

#-------------------------------------------------------------------------------
class COutputRow:
  def __init__(self):
    self.good = 0
    self.name = None
    self.values = OrderedDict()
  
  def __str__(self):
    return f"Good {self.good} Name {self.name} Values {self.values}"

  def __repr__(self):
    return self.__str__()

#-------------------------------------------------------------------------------
def compare_list(value1 : str, value2 : str) -> float:
  s1 = set( json.loads(value1) )
  s2 = set( json.loads(value2) )
  val = 0.0
  if len(s1) == 0 or len(s2) == 0:
    val = INVALID_SCORE
  else:
    inter = len(s1.intersection(s2))
    maxs  = len(max(s1, s2))
    val = (inter * 100) / maxs
    val /= 100
  return val

#-------------------------------------------------------------------------------
def compare_row(d : dict, same_binary : bool) -> COutputRow:

  test_ratio = 0.0
  total_fields = 0

  out = COutputRow()
  out.good = same_binary and (d["ea1"] == d["ea2"] or d["name1"] == d["name2"])
  out.name = f'{d["name1"]} - {d["name2"]}'
  out.values["name"] = int(out.good)
  out.values["ratio"] = 0

  for field in FIELDS:
    if field == "name":
      continue

    val1 = d[f"{field}1"]
    val2 = d[f"{field}2"]

    tmp = 0.0
    if type(val1) is int and type(val2) is int:
      tmp = int_compare_ratio(val1, val2)
    elif type(val1) is str and type(val2) is str:
      if val1.startswith("["):
        tmp = compare_list(val1, val2)
      else:
        tmp = quick_ratio(val1, val2)
    elif val1 is None or val2 is None:
      tmp = INVALID_VALUE
    else:
      raise Exception("wut?")

    total_fields += 1
    test_ratio += tmp

    out.values[field] = tmp
    if field in NUM_FIELDS:
      v1 = val1 if val1 is not None else INVALID_VALUE
      v2 = val2 if val2 is not None else INVALID_VALUE
      out.values[f"min_{field}"] = min(v1, v2)
      out.values[f"max_{field}"] = max(v1, v2)

  out.values["ratio"] = test_ratio / total_fields
  return out

#-------------------------------------------------------------------------------
class CDatasetBuilder:
  def __init__(self):
    self.samples = []

    self.current_sample1 = None
    self.current_sample2 = None

    self.print_lock = threading.Lock()
    self.verbose = False

    self.output_file = None
    self.lines_buffer = []

  def __del__(self):
    if self.output_file is not None:
      self.output_file.close()

  def log(self, msg : str):
    log(msg)
  
  def debug(self, msg: str):
    debug(msg)

  def exec_cursor(self, db : sqlite3.Connection, sql : str, parameters : tuple[()]=()) -> sqlite3.Cursor:
    cur = db.cursor()
    cur.execute(sql, parameters)
    return cur

  def get_cpu_for_target(self, filename : str) -> str:
    ret = ""
    db = sqlite3_connect(filename)
    try:
      sql = "select processor from program"
      cur = self.exec_cursor(db, sql)
      try:
        cur.execute(sql)
        row = cur.fetchone()
        if row is not None:
          ret = row[0]
      finally:
        cur.close()
    finally:
      db.close()
    return ret

  def process(self, filename : str):
    d = {}
    try:
      cpu = self.get_cpu_for_target(filename)
    except:
      log(f"Error reading CPU for database {filename}: {sys.exc_info()[1]}")
      return False

    d["cpu"]  = cpu
    d["arch"] = cpu
    d["name"] = os.path.basename(filename)
    d["path"] = filename
    self.samples.append(d)

  def search(self, directory : str, filter_str : str):
    for root, dirs, files in os.walk(directory, topdown=False):
      for name in files:
        if not name.endswith(".sqlite"):
          continue

        if filter_str is None or re.match(filter_str, name):
          filename = os.path.join(root, name)
          if self.verbose:
            self.debug(f"Processing filename {filename}")
          self.process(filename)
    if self.verbose: self.debug(f"Total of {len(self.samples)} sample(s) to process")

  def is_same_binary(self, db : sqlite3.Connection) -> bool:
    ret = False
    sql = """ select 1 from main.program p, diff.program dp where p.md5sum = dp.md5sum"""
    cur = self.exec_cursor(db, sql)
    try:
      cur.execute(sql)
      row = cur.fetchone()
      ret = row is not None

      if not ret:
        total = 0
        sql = "select count(0) from main.functions"
        cur.execute(sql)
        row = cur.fetchone()
        total = row[0]

        sql = """select count(0)
                  from main.functions f,
                        diff.functions df
                  where f.address = df.address """
        cur.execute(sql)
        row = cur.fetchone()
        matches = row[0]
        percent = (matches * 100) / total
        if percent >= SAME_BINARY_PERCENT:
          ret = True
        
        if not ret:
          sql = """select count(0)
                     from main.functions f,
                          diff.functions df
                    where f.name = df.name """
          cur.execute(sql)
          row = cur.fetchone()
          matches = row[0]
          percent = (matches * 100) / total
          if percent >= SAME_NAMES_PERCENT:
            ret = True
    except:
      debug(f"Error: {sys.exc_info()[1]}")
      ret = "ERROR"
    finally:
      cur.close()
    return ret

  def process_single_row(self, d : dict, same_binary : bool):
    ret = compare_row(d, same_binary)
    l = list(map(str, ret.values.values()))
    l.insert(0, str(int(self.current_sample1["cpu"] == self.current_sample2["cpu"])))
    l.insert(1, str(int(self.current_sample1["arch"] == self.current_sample2["arch"])))
    line = CSV_SEPARATOR_FIELD.join(l)

    self.print_lock.acquire()
    self.lines_buffer.append(line)
    #print(line)
    self.print_lock.release()

  def generate_false_matches(self, db : sqlite3.Connection, d : dict, total : int, same_binary : bool):
    sql = DIFFERENT_SAMPLES_SQL
    cur = self.exec_cursor(db, sql, (d["id1"], total,))
    while 1:
      row = cur.fetchone()
      if row is None:
        break
      d = dict(row)
      self.process_single_row(d, same_binary)
    cur.close()

  def compare_samples(self, db : sqlite3.Connection, sample1 : dict, sample2 : dict, same_binary : bool):
    cur = db.cursor()
    try:
      # Find all the good matches
      sql = COMPARE_SAMPLES_SQL
      cur.execute(sql)

      while 1:
        row = cur.fetchone()
        if not row:
          break

        d = dict(row)
        if d["name1"].startswith(".") or d["name2"].startswith("."):
          continue

        self.process_single_row(d, same_binary)
        self.generate_false_matches(db, d, 5, same_binary)
    finally:
      cur.close()

  def check_files_same_binary(self, sample1 : dict, sample2 : dict):
    path1 = os.path.abspath(sample1["path"])
    path2 = os.path.abspath(sample2["path"])
    db = sqlite3_connect(path1)
    db.execute(f'attach \"{path2}\" as diff')
    same_binary = self.is_same_binary(db)
    if same_binary == "ERROR":
      debug(f"Error with database {path1} or {path2}")
    db.close()
    return same_binary

  def build_dataset_row(self, sample1 : dict, sample2 : dict):
    path1 = os.path.abspath(sample1["path"])
    path2 = os.path.abspath(sample2["path"])
    db = sqlite3_connect(path1)
    db.execute(f'attach \"{path2}\" as diff')
    same_binary = self.is_same_binary(db)
    if same_binary == "ERROR":
      debug(f"Error with database {path1} or {path2}")
    elif same_binary:
      debug(f"Found same binaries {sample1['path']} - {sample2['path']} Same binary? {self.is_same_binary(db)}")
      ret = self.compare_samples(db, sample1, sample2, same_binary)
    db.close()

  def flush(self):
    self.output_file.write("%s\n" % "\n".join(self.lines_buffer))
    self.lines_buffer.clear()

  def build(self):
    if os.getenv("NO_PRINT_HEADER") is None:
      line = "cpu;arch;function;" + CSV_SEPARATOR_FIELD.join(PRINT_FIELDS)
      line = line.replace(";", CSV_SEPARATOR_FIELD)
      self.output_file.write("%s\n" % line)

    q = list()
    cpus = cpu_count()
    t = time.monotonic()

    samples_done = set()
    print_flag = 1
    dones = 0
    total = (len(self.samples) * len(self.samples)) - len(self.samples)
    for sample1 in self.samples:
      for sample2 in self.samples:
        dones += 1
        if sample1 == sample2:
          continue

        item = f"{sample1}-{sample2}"
        if item in samples_done:
          continue
        samples_done.add(item)

        item = f"{sample2}-{sample1}"
        if item in samples_done:
          continue
        samples_done.add(item)

        self.current_sample1 = sample1
        self.current_sample2 = sample2

        line1 = f"""{sample1["cpu"]} {sample1["name"]}"""
        line2 = f"""{sample2["cpu"]} {sample2["name"]}"""

        if self.verbose: self.debug(f"Comparing samples {line1} - {line2}")

        if self.check_files_same_binary(sample1, sample2):
          q.append([sample1, sample2])

        # ret = self.build_dataset_row(sample1, sample2)

        if len(q) >= cpus or len(q) == total:
          print_flag += 1
          wait_list = list()
          for item in q:
            th = threading.Thread(target=self.build_dataset_row, args=[item[0], item[1]])
            th.start()
            wait_list.append(th)

          for th in wait_list:
            th.join()
            if self.verbose: self.debug(f"Thread {th} done...")

          q.clear()
          self.flush()

      if print_flag % 10 == 0:
        elapsed   = time.monotonic() - t
        remaining = (elapsed / dones) * (total - dones)
        m, s = divmod(remaining, 60)
        h, m = divmod(m, 60)
        m_elapsed, s_elapsed = divmod(elapsed, 60)
        h_elapsed, m_elapsed = divmod(m_elapsed, 60)
        self.debug(f"Total of {dones} samples processed out of {total}")
        self.debug(f"Elapsed %d:%02d:%02d second(s), remaining time ~%d:%02d:%02d" % (h_elapsed, m_elapsed, s_elapsed, h, m, s))
    
    wait_list = list()
    for item in q:
      th = threading.Thread(target=self.build_dataset_row, args=[item[0], item[1]])
      th.start()
      wait_list.append(th)

    for th in wait_list:
      th.join()
      if self.verbose: self.debug(f"Thread {th} done...")

    q.clear()
    self.flush()

    elapsed   = time.monotonic() - t
    remaining = (elapsed / dones) * (total - dones)
    m, s = divmod(remaining, 60)
    h, m = divmod(m, 60)
    m_elapsed, s_elapsed = divmod(elapsed, 60)
    h_elapsed, m_elapsed = divmod(m_elapsed, 60)
    self.debug(f"Total of {dones} samples processed out of {total}")
    self.debug(f"Elapsed %d:%02d:%02d second(s)" % (h_elapsed, m_elapsed, s_elapsed))


#-------------------------------------------------------------------------------
def build_dataset(args : argparse.Namespace):
  builder = CDatasetBuilder()
  builder.output_file = open(args.filename, "w")
  builder.verbose = args.verbose
  builder.search(args.directory, args.filter)
  out = builder.build()

#-------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(prog=sys.argv[0],
                    description="Create a dataset out of multiple Diaphora exported *.sqlite files",
                    epilog="Copyright (c) 2015-2024, Joxean Koret")
  parser.add_argument("directory", help="Directory with Diaphora exported *.sqlite databases")
  parser.add_argument("-o", dest="filename", help="Output .csv file", required=True)
  parser.add_argument("-f", "--filter", help="Filter to apply when searching *.sqlite database files")
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()
  if args.filename is None:
    print("No output filename specified")
    sys.exit(1)
  build_dataset(args)

if __name__ == "__main__":
  main()

