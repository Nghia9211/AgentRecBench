@echo off
setlocal

:: Thiet lap thu muc lam viec
set work_dir=.
cd %work_dir%

:: Dinh nghia cac bien
set "DATASET=yelp" 

set DATA_DIR=./data/%DATASET%/
set STAGE=test
set CANS_NUM=20
set MAX_SAMPLES=100
set MAX_EPOCH=5
set MODEL=gpt-3.5-turbo
set API_KEY=""
set MODEL_PATH=./saved_models/%DATASET%_best_model.pt
set CANDIDATE_DIR="C:\Users\Admin\Desktop\Document\AgenticCode\AgentRecBench\dataset\task\user_cold_start\%DATASET%\tasks"
set ITEMFILE="C:\Users\Admin\Desktop\Document\AgenticCode\AgentRecBench\dataset\output_data_all\item.json"
set MAX_RETRY_NUM=5
set SEED=303
set MP=1
set TEMPERATURE=0.0
set LABEL=dialogue_from_sasrec
set P_MODEL=SASRec


set INPUT_JSON_FILE=./final_mask_%DATASET%.json 

set HIDDEN_SIZE=64
set DROPOUT=0.1

set OUTPUT_FILE=./output/%DATASET%_rec_eval/%P_MODEL%_%LABEL%_%MODEL%_%SEED%_%TEMPERATURE%_%MP%_%MAX_EPOCH%.jsonl

set SAVE_REC_DIR=./output/%DATASET%_rec_save/rec_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%
set SAVE_USER_DIR=./output/%DATASET%_rec_save/user_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%

set RESULT_FILE=./output/%DATASET%_result/evaluation_results_AFL_%DATASET%.json

:: In thong tin ra man hinh
echo DATA_DIR = %DATA_DIR%
echo MODEL_PATH = %MODEL_PATH%
echo INPUT_JSON_FILE = %INPUT_JSON_FILE%
echo ITEMFILE = %ITEMFILE%  
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
echo HIDDEN_SIZE = %HIDDEN_SIZE% 
echo DROPOUT = %DROPOUT% 
echo CANDIDATE_DIR = %CANDIDATE_DIR%
echo OUTPUT_FILE = %OUTPUT_FILE%
echo SAVE_REC_DIR = %SAVE_REC_DIR%
echo SAVE_USER_DIR = %SAVE_USER_DIR%
echo RESULT_FILE = %RESULT_FILE%

:: Thiet lap GPU
set CUDA_VISIBLE_DEVICES=0

:: Chay lenh Python
python ./main.py ^
    --data_dir="%DATA_DIR%" ^
    --model_path="%MODEL_PATH%" ^
    --input_json_file="%INPUT_JSON_FILE%" ^
    --stage="%STAGE%" ^
    --cans_num=%CANS_NUM% ^
    --max_epoch=%MAX_EPOCH% ^
    --max_samples=%MAX_SAMPLES% ^
    --output_file="%OUTPUT_FILE%" ^
    --model="%MODEL%" ^
    --api_key="%API_KEY%" ^
    --candidate_dir=%CANDIDATE_DIR% ^
    --max_retry_num=%MAX_RETRY_NUM% ^
    --seed=%SEED% ^
    --item_mapping_file=%ITEMFILE% ^
    --mp=%MP% ^
    --temperature=%TEMPERATURE% ^
    --hidden_size=%HIDDEN_SIZE% ^
    --dropout=%DROPOUT% ^
    --save_info ^
    --save_rec_dir="%SAVE_REC_DIR%" ^
    --save_user_dir="%SAVE_USER_DIR%" ^
    --result_file=%RESULT_FILE%

endlocal
pause