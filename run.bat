@echo off
setlocal enabledelayedexpansion

@REM 
if exist .env (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else if exist baseline\.env (
    for /f "usebackq tokens=1,2 delims==" %%a in ("baseline\.env") do (
        set "%%a=%%b"
    )
) else (
    echo [ERROR] Khong tim thay file .env!
    pause
    exit /b
)

echo =====================================================
echo 1. CHON DATASET DE CHAY SIMULATION
echo =====================================================
echo [1] amazon
echo [2] goodreads
echo [3] yelp
echo.
set /p ds_choices="Nhap lua chon dataset (Vi du: 1 2): "

echo.
echo =====================================================
echo 2. CHON SCENARIO DE CHAY SIMULATION
echo =====================================================
echo [1] classic
echo [2] user_cold_start
echo [3] item_cold_start
echo.
set /p sc_choices="Nhap lua chon scenario (Vi du: 1 2 3): "

echo.
set /p num_runs="Nhap so lan lap moi thi nghiem (Vi du: 5): "

for %%d in (%ds_choices%) do (
    set "DS="
    if "%%d"=="1" set "DS=amazon"
    if "%%d"=="2" set "DS=goodreads"
    if "%%d"=="3" set "DS=yelp"

    if defined DS (
        for %%s in (%sc_choices%) do (
            set "SC="
            if "%%s"=="1" set "SC=classic"
            if "%%s"=="2" set "SC=user_cold_start"
            if "%%s"=="3" set "SC=item_cold_start"

            if defined SC (
                echo.
                echo [STARTING] Dataset: !DS! ^| Scenario: !SC!
                
                @REM Vong lap chay n lan de lay ket qua trung binh
                for /L %%r in (1,1,%num_runs%) do (
                    echo.
                    echo [RUN %%r/%num_runs%] Dataset: !DS! ^| Scenario: !SC!
                    
                    @REM Cap nhat duong dan baseline/ cho cac file python
                    @REM python baseline/CoTAgent_baseline.py --task_set !DS! --scenario !SC! 
                    @REM python baseline/CoTMemoryAgent_baseline.py --task_set !DS! --scenario !SC!
                    @REM python baseline/MemoryAgent_baseline.py --task_set !DS! --scenario !SC! 
                    @REM python baseline/DummyAgent_baseline.py --task_set !DS! --scenario !SC! 
                    @REM python baseline/RecHackerAgent_baseline.py --task_set !DS! --scenario !SC!  
                    @REM python baseline/Baseline666_baseline.py --task_set !DS! --scenario !SC! 
                    
                    @REM Currently Run On Server for Test Result
                    @REM python baseline/ARAGAgent.py --task_set !DS! --scenario !SC! 
                    
                    @REM python baseline/ARAGgcnAgent.py --task_set !DS! --scenario !SC!
                    
                    @REM python baseline/ARAGAgent_init.py --task_set !DS! --scenario !SC! 
                    
                    python baseline/ARAGgcnAgentRetrie.py --task_set !DS! --scenario !SC! 
                    
                    @REM ---------------------------------------

                    @REM --- LOGIC DOI TEN FILE TU DONG ---
                    @REM Tim file .json trong thu muc baseline/results/!SC!/
                    
                    set "TARGET_DIR=./baseline/results/!SC!"
                    set "FOUND_FILE="
                    
                    for /f "delims=" %%f in ('dir /b "!TARGET_DIR!\evaluation_results_*_!DS!.json" 2^>nul') do (
                        set "FILENAME=%%f"
                        @REM Kiem tra neu file chua co hau to _run thi moi doi ten
                        echo !FILENAME! | findstr /v "_run" >nul
                        if !errorlevel! == 0 (
                            set "FOUND_FILE=%%f"
                        )
                    )

                    if defined FOUND_FILE (
                        set "OLD_PATH=!TARGET_DIR!\!FOUND_FILE!"
                        @REM Tach ten file va duoi file
                        for %%A in ("!OLD_PATH!") do (
                            set "NAME_ONLY=%%~nA"
                            set "EXT_ONLY=%%~xA"
                        )
                        set "NEW_PATH=!TARGET_DIR!\!NAME_ONLY!_run%%r!EXT_ONLY!"
        
                        move /y "!OLD_PATH!" "!NEW_PATH!"
                    ) else (
                        echo [WARNING] Khong tim thay file ket qua moi tai !TARGET_DIR! de doi ten.
                    )
                }
            )
        )
    )
)

echo.
echo =====================================================
echo Tat ca cac thi nghiem da hoan thanh.
echo =====================================================
pause