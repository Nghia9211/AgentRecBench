
@echo off
setlocal
set "PYTHON_SCRIPT=build_vector_db.py"
set "EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2"
set "BATCH_SIZE=256"
set "DATA_DIR=../data/graph_data"
set "STORAGE_DIR=../storage"


echo ======================================================
echo Building vector store for item...
echo ======================================================

python %PYTHON_SCRIPT% ^
    --data_path "%DATA_DIR%/item_yelp.json" ^
    --save_path "%STORAGE_DIR%/item_storage_yelp" ^
    --embed_model "%EMBED_MODEL%" ^
    --batch_size %BATCH_SIZE%

echo.
echo User review database created successfully in '%STORAGE_DIR%/user_storage'.

echo.
echo ======================================================
echo Script finished.
echo ======================================================

endlocal
pause