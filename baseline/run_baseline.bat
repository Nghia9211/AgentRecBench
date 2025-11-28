@echo off
setlocal

set "DATASET=yelp"


echo Select input : Amazon, Goodreads, Yelp
echo Starting Simulation 
@REM python CoTAgent_baseline.py --task_set %DATASET%
python CoTMemoryAgent_baseline.py --task_set %DATASET%
@REM python MemoryAgent_baseline.py --task_set %DATASET%
@REM python DummyAgent_baseline.py --task_set %DATASET%
@REM python RecHackerAgent_baseline.py --task_set %DATASET%
@REM python ARAGAgent_baseline.py --task_set %DATASET%
@REM python ARAGgcnAgent_baseline.py --task_set %DATASET%
pause