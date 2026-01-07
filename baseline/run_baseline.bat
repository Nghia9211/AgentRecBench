@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo CHON DATASET DE CHAY SIMULATION
echo =====================================================
echo [1] amazon
echo [2] goodreads
echo [3] yelp
echo.
echo (Vi du: Nhap "1 2" de chay Amazon va Goodreads cung luc)
echo =====================================================
set /p choices="Nhap lua chon cua ban: "

for %%c in (%choices%) do (
    set "DS="
    if "%%c"=="1" set "DS=amazon"
    if "%%c"=="2" set "DS=goodreads"
    if "%%c"=="3" set "DS=yelp"

    if defined DS (
        echo Dang bat dau simulation cho: !DS!
        @REM python CoTAgent_baseline.py --task_set !DS! & ^
        @REM python CoTMemoryAgent_baseline.py --task_set !DS! & ^
        @REM python MemoryAgent_baseline.py --task_set !DS! & ^
        @REM python DummyAgent_baseline.py --task_set !DS! & ^
        @REM python RecHackerAgent_baseline.py --task_set !DS! & ^
        @REM python ARAGAgent_baseline.py --task_set !DS! & ^
        python ARAGgcnAgent_baseline.py --task_set !DS! & ^
        @REM python ARAGAgent_init_baseline.py --task_set !DS! & ^
        @REM python ARAGgcnAgentRetrie_baseline.py --task_set !DS! & ^
    )
)

echo.
echo =====================================================
echo Cac tien trinh da duoc kich hoat. Kiem tra cac cua so moi.
echo =====================================================
pause