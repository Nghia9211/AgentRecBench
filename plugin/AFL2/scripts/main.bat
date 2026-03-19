@echo off
setlocal

:: Thiet lap thu muc lam viec
set work_dir=.
cd %work_dir%

:: Dinh nghia cac bien
set "DATASET=amazon" 

set DATA_DIR=./data/%DATASET%/
set STAGE=test
set CANS_NUM=20
set MAX_SAMPLES=100
set MAX_EPOCH=5
set MODEL=gpt-3.5-turbo
set API_KEY=your_key_here
set MAX_RETRY_NUM=5
set SEED=303
set MP=1
set TEMPERATURE=0.0
set LABEL=recommendation_experiment
set P_MODEL=SASRec
set PRIOR_FILE=./data/%DATASET%/%DATASET%_prior.csv

:: Tao ten file output dua tren cac bien tren
set OUTPUT_FILE=./output/%DATASET%_rec_eval/%P_MODEL%_%LABEL%_%MODEL%_%SEED%_%TEMPERATURE%_%MP%_%MAX_EPOCH%.jsonl

set SAVE_REC_DIR=./output/%DATASET%_rec_save/rec_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%
set SAVE_USER_DIR=./output/%DATASET%_rec_save/user_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%

:: In thong tin ra man hinh
echo DATA_DIR = %DATA_DIR%
echo MODEL_PATH = %MODEL_PATH%
echo PRIOR_FILE = %PRIOR_FILE%
echo STAGE = %STAGE%
echo CANS_NUM = %CANS_NUM%
echo MAX_EPOCH = %MAX_EPOCH%
echo MAX_SAMPLES = %MAX_SAMPLES%
echo MODEL = %MODEL%
echo API_KEY = %API_KEY%
echo MAX_RETRY_NUM = %MAX_RETRY_NUM%
echo SEED = %SEED%
echo TEMPERATURE = %TEMPERATURE%
echo MP = %MP%
echo OUTPUT_FILE = %OUTPUT_FILE%
echo SAVE_REC_DIR = %SAVE_REC_DIR%
echo SAVE_USER_DIR = %SAVE_USER_DIR%

:: Thiet lap GPU
set CUDA_VISIBLE_DEVICES=0

:: Chay lenh Python
python ./main.py ^
    --data_dir=%DATA_DIR% ^
    --model_path=%MODEL_PATH% ^
    --prior_file=%PRIOR_FILE% ^
    --stage=%STAGE% ^
    --cans_num=%CANS_NUM% ^
    --max_epoch=%MAX_EPOCH% ^
    --max_samples=%MAX_SAMPLES% ^
    --output_file=%OUTPUT_FILE% ^
    --model=%MODEL% ^
    --api_key=%API_KEY% ^
    --max_retry_num=%MAX_RETRY_NUM% ^
    --seed=%SEED% ^
    --mp=%MP% ^
    --temperature=%TEMPERATURE% ^
    --save_info ^
    --save_rec_dir=%SAVE_REC_DIR% ^
    --save_user_dir=%SAVE_USER_DIR%

endlocal
pause