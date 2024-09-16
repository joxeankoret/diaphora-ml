#!/bin/bash

# Change to your IDA binary
IDA=/home/joxean/ida90beta/idat
# Change to your Diaphora script
DIAPHORA=/home/joxean/Documents/research/diaphora/public/diaphora.py

if [ $# -eq 0 ]; then
  echo "Usage: $0 <sample>"
  exit 1
fi

if [[ -f $1 ]]; then
  export SAMPLE=$1
  export DIAPHORA_AUTO=1
  export DIAPHORA_USE_DECOMPILER=1
  export DIAPHORA_EXPORT_FILE=$1.sqlite

  echo "[`date` $$] Analysing ${SAMPLE} ..."
  ${IDA} -A -B -S${DIAPHORA} -L${SAMPLE}.log ${SAMPLE}
  echo "[`date` $$] Done"
else
  echo "File $1 does not exist!"
  exit 2
fi

