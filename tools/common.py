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
import sqlite3

try:
  from cdifflib import CSequenceMatcher as SequenceMatcher # type: ignore
except ImportError:
  from difflib import SequenceMatcher

#-------------------------------------------------------------------------------
_DEBUG = False
def debug(msg : str):
  sys.stderr.write(f"[Diaphora: {time.asctime()}] {msg}\n")
  sys.stderr.flush()

#-------------------------------------------------------------------------------
def log(message : str):
  print(f"[Diaphora: {time.asctime()} {os.getpid()}] {message}", flush=True)

#-------------------------------------------------------------------------------
def sqlite3_connect(db_name):
  """
  Return a SQL connection object.
  """
  db = sqlite3.connect(db_name, check_same_thread=False)
  db.text_factory = str
  db.row_factory = sqlite3.Row
  return db

#-------------------------------------------------------------------------------
def int_compare_ratio(value1 : int, value2 : int) -> float:
  """
  Get a similarity ratio for two integers.
  """
  if value1 + value2 == 0:
    val = 1.0
  else:
    val = 1 - ( abs(value1 - value2) / max(value1, value2) )
  return val

#-------------------------------------------------------------------------------
def quick_ratio(buf1 : str, buf2 : str) -> float:
  """
  Call SequenceMatcher.quick_ratio() to get a comparison ratio.
  """
  if buf1 is None or buf2 is None or buf1 == "" or buf1 == "":
    return 0

  if buf1 == buf2:
    return 1.0

  s1 = buf1.lower().split('\n')
  s2 = buf2.lower().split('\n')
  seq = SequenceMatcher(None, s1, s2)
  return seq.ratio()
