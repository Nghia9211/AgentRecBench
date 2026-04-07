@echo off
setlocal enabledelayedexpansion

@REM K
if exist .env (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else if exist baseline\.env (
    for /f "usebackq tokens=1,2 delims==" %%a in ("baseline\.env") do (
        set "%%a=%%b"
    )
) else (
    echo [ERROR] Not Found .env!
    pause
    exit /b
)

echo =====================================================
echo 1. Choose Dataset to Run Simulator
echo =====================================================
echo [1] amazon
echo [2] goodreads
echo [3] yelp
echo.
set /p ds_choices="Select Dataset (Ex: 1 2): "

echo.
echo =====================================================
echo 2. CHON SCENARIO DE CHAY SIMULATION
echo =====================================================
echo [1] classic
echo [2] user_cold_start
echo [3] item_cold_start
echo.
set /p sc_choices="Select scenario (Ex: 1 2 3): "

echo.
echo =====================================================
echo 3. CHON LLM PROVIDER
echo =====================================================
echo [1] ollama
echo [2] gpt
echo [3] groq
echo.
set /p prov_choice="Select Provider (1, 2, or 3): "
set "PROV="
if "%prov_choice%"=="1" set "PROV=ollama"
if "%prov_choice%"=="2" set "PROV=gpt"
if "%prov_choice%"=="3" set "PROV=groq"

echo.
set /p MODEL_NAME="Enter Model Name (Leave blank to use default): "

echo.
set /p num_runs="Enter number of experiments to run : (Ex: 5): "

pushd baseline

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
                echo [STARTING] Dataset: !DS! ^| Scenario: !SC! ^| Provider: !PROV!
                
                for /L %%r in (1,1,%num_runs%) do (
                    echo.
                    echo [RUN %%r/%num_runs%] Dataset: !DS! ^| Scenario: !SC!

                    @REM Thiết lập tham số model nếu có nhập
                    set "MODEL_ARG="
                    if not "!MODEL_NAME!"=="" set "MODEL_ARG=--model !MODEL_NAME!"

                    @REM python CoTAgent_baseline.py --task_set !DS! --scenario !SC! --provider !PROV! !MODEL_ARG!
                    @REM python CoTMemoryAgent_baseline.py --task_set !DS! --scenario !SC! --provider !PROV! !MODEL_ARG!
                    @REM python MemoryAgent_baseline.py --task_set !DS! --scenario !SC! --provider !PROV! !MODEL_ARG!
                    @REM python DummyAgent_baseline.py --task_set !DS! --scenario !SC! --provider !PROV! !MODEL_ARG!
                    @REM python RecHackerAgent_baseline.py --task_set !DS! --scenario !SC! --provider !PROV! !MODEL_ARG!
                    python Baseline666_baseline.py --task_set !DS! --scenario !SC! 
                    
                    @REM Currently Run On Server for Test Result
                    @REM python ARAGgcnAgentRetrie.py --task_set !DS! --scenario !SC! 

                    set "TARGET_DIR=./results/!SC!"
                    set "FOUND_FILE="
                    
                    for /f "delims=" %%f in ('dir /b "!TARGET_DIR!\evaluation_results_*_!DS!.json" 2^>nul') do (
                        set "FILENAME=%%f"
                        echo !FILENAME! | findstr /v "_run" >nul
                        if !errorlevel! == 0 (
                            set "FOUND_FILE=%%f"
                        )
                    )

                    if defined FOUND_FILE (
                        set "OLD_PATH=!TARGET_DIR!\!FOUND_FILE!"
                        for %%A in ("!OLD_PATH!") do (
                            set "NAME_ONLY=%%~nA"
                            set "EXT_ONLY=%%~xA"
                        )
                        set "NEW_PATH=!TARGET_DIR!\!NAME_ONLY!_run%%r!EXT_ONLY!"
        
                        move /y "!OLD_PATH!" "!NEW_PATH!"
                    ) else (
                        echo [WARNING] Not Found Result File to change name.
                    )
                )
            )
        )
    )
)

@REM 
popd

echo.
echo =====================================================
echo Tat ca cac thi nghiem da hoan thanh.
echo =====================================================
pause