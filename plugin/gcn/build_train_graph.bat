@echo off
TITLE LightGCN Pipeline (Build Graph + Train)


set BUILD_SCRIPT=build_graph_lgcn.py
set TRAIN_SCRIPT=train.py

set DATASET=yelp

set BASE_PATH=graph_data
set USER_FILE=%BASE_PATH%\user_%DATASET%.json
set ITEM_FILE=%BASE_PATH%\item_%DATASET%.json
set REVIEW_FILE=%BASE_PATH%\review_%DATASET%.json
set GT_FILE=%BASE_PATH%\ground_truth_gcn_%DATASET%.json

set OUTPUT_DATA=gcn_graph\processed_graph_data_%DATASET%.pt
set OUTPUT_EMB=gcn_embedding\gcn_embeddings_3hop_%DATASET%.pt

echo.
echo ========================================================
echo        STEP 1: STARTING GRAPH BUILD PROCESS
echo ========================================================
echo Input User:   "%USER_FILE%"
echo Output Data:  "%OUTPUT_DATA%"
echo.


python %BUILD_SCRIPT% ^
    --user_file "%USER_FILE%" ^
    --item_file "%ITEM_FILE%" ^
    --review_file "%REVIEW_FILE%" ^
    --gt_folder "%GT_FILE%" ^
    --output_file "%OUTPUT_DATA%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build Graph Failed! Check if files exist.
    pause
    exit /b
)

echo [SUCCESS] Graph built successfully!

:: --- 3. TRAINING PHASE ---

set EPOCHS=450
set BATCH_SIZE=1024
set LR=0.01
set REG=1e-4
set DEVICE=cuda

echo.
echo ========================================================
echo        STEP 2: STARTING TRAINING MODEL
echo ========================================================
echo Settings: Epochs=%EPOCHS% - Emb=%EMB_DIM% - Device=%DEVICE%
echo.

python %TRAIN_SCRIPT% ^
    --data_file "%OUTPUT_DATA%" ^
    --export_file "%OUTPUT_EMB%" ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --reg %REG% ^
    --device %DEVICE% ^
    --dataset %DATASET%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed or interrupted!
    pause
    exit /b
)

