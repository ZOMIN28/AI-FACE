@echo off
rem Activate conda environment
call conda activate AI_FACE

python main.py

rem Close conda environment
call conda deactivate
