@echo off
setlocal

set work_dir=.
cd %work_dir%

@REM set "dataset=goodreads"
@REM python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100

set "dataset=amazon"
python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100

@REM set "dataset=yelp"
@REM python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100


