@echo off
setlocal

set work_dir=.
cd %work_dir%

set "dataset=goodreads"
python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100

@REM set "dataset=amazon"
@REM python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100

@REM set "dataset=yelp"
@REM python ./utils/train_sasrec.py --data_dir ./data/%dataset% --epochs 100


