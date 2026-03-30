
@echo off
setlocal
set "PYTHON_SCRIPT=build_vector_db.py"
set "EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2"
set "BATCH_SIZE=256"
set "DATA_DIR=../gcn/graph_data"
set "STORAGE_DIR=../storage"
set "DATASET=yelp"

echo ======================================================
echo Building vector store for item...
echo ======================================================

python %PYTHON_SCRIPT% ^
    --data_path "%DATA_DIR%/item_%DATASET%.json" ^
    --save_path "%STORAGE_DIR%/item_storage_%DATASET%" ^
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