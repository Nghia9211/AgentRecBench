@echo off
setlocal

:: ============================================================
::  AFL + ARAG Integration Run Script (Amazon Cold-Start)
:: ============================================================

set work_dir=.
cd %work_dir%

:: --- Dataset ---
set "DATASET=goodreads"
set "SCENARIO=user_cold_start"

set DATA_DIR=./data/%DATASET%/
set STAGE=test
set CANS_NUM=20
set MAX_SAMPLES=100
set MAX_EPOCH=5
set MODEL=gpt-3.5-turbo
set API_KEY="YOURAPIKEYHERE"
set MODEL_PATH=./saved_models/%DATASET%_best_model.pt
set CANDIDATE_DIR="..\..\dataset\task\%SCENARIO%\%DATASET%\tasks"
set ITEMFILE="..\..\dataset\output_data_all\item.json"
set MAX_RETRY_NUM=5
set SEED=303
set MP=1
set TEMPERATURE=0.0
set INPUT_JSON_FILE=./ground_truth.json
set HIDDEN_SIZE=64

:: --- ARAG-specific paths ---
set USE_ARAG=--use_arag
set FAISS_DB_PATH="..\storage\item_storage_%DATASET%"
set GCN_PATH="..\gcn\gcn_embedding\gcn_embeddings_3hop_%DATASET%.pt"
set NLI_THRESHOLD=5.5
set EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

set LABEL=AFL_ARAG_coldstart
set P_MODEL=SASRec


set OUTPUT_FILE=./output/%DATASET%_%SCENARIO%_rec_eval/%P_MODEL%_%LABEL%_%MODEL%_%SEED%_%TEMPERATURE%_%MP%_%MAX_EPOCH%.jsonl
set SAVE_REC_DIR=./output/%DATASET%_%SCENARIO%_rec_save/rec_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%
set SAVE_USER_DIR=./output/%DATASET%_%SCENARIO%_rec_save/user_%P_MODEL%_%LABEL%_%MODEL%_%MAX_EPOCH%
set RESULT_FILE=../../baseline/results/%SCENARIO%/evaluation_results_AFL_ARAG_%DATASET%.json

:: --- Print config ---
echo ============================================================
echo   AFL + ARAG Integration
echo   Dataset:       %DATASET%
echo   FAISS DB:      %FAISS_DB_PATH%
echo   GCN Path:      %GCN_PATH%
echo   NLI Threshold: %NLI_THRESHOLD%
echo   Max Epoch:     %MAX_EPOCH%
echo   Max Samples:   %MAX_SAMPLES%
echo ============================================================

set CUDA_VISIBLE_DEVICES=0

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
    --api_key=%API_KEY% ^
    --candidate_dir=%CANDIDATE_DIR% ^
    --max_retry_num=%MAX_RETRY_NUM% ^
    --seed=%SEED% ^
    --item_mapping_file=%ITEMFILE% ^
    --mp=%MP% ^
    --temperature=%TEMPERATURE% ^
    --hidden_size=%HIDDEN_SIZE% ^
    --save_info ^
    --save_rec_dir="%SAVE_REC_DIR%" ^
    --save_user_dir="%SAVE_USER_DIR%" ^
    --result_file=%RESULT_FILE% ^
    %USE_ARAG% ^
    --faiss_db_path=%FAISS_DB_PATH% ^
    --gcn_path=%GCN_PATH% ^
    --nli_threshold=%NLI_THRESHOLD% ^
    --embed_model_name="%EMBED_MODEL%"

endlocal
pause