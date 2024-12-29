@echo off
ECHO Starting Batch Processing...
ECHO ===============================

:: Activate conda environment
call conda activate metatransformers

:: Set paths
set BASE_DIR=%~dp0
set BIONN_DIR=%BASE_DIR%bioNN

:: Get batch parameters
set /p BATCH_SIZE="Enter batch size (default=10): " || set BATCH_SIZE=10
set /p ITERATIONS="Enter number of iterations (default=100): " || set ITERATIONS=100

:: Run batch processor
ECHO Running batch process...
python %BIONN_DIR%\run_logic_batch.py %BATCH_SIZE% %ITERATIONS%

ECHO ===============================
ECHO Batch processing complete
ECHO Check logic_batch.log for details

:: Keep window open
pause