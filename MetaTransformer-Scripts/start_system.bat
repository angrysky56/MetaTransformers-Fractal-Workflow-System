@echo off
ECHO Starting MetaTransformers System...
ECHO ===============================

:: Activate conda environment
ECHO Activating metatransformers environment...
call conda activate metatransformers

:: Set paths
set BASE_DIR=%~dp0
set BIONN_DIR=%BASE_DIR%bioNN
set LOGIC_DIR=%BIONN_DIR%\logic_processing
set LOGIC_LLM_DIR=%LOGIC_DIR%\Logic-LLM

:: Check environment
python -c "import torch" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    ECHO Error: PyTorch not found in environment
    ECHO Please ensure all dependencies are installed
    exit /b 1
)

:: Initialize Neo4j if needed
ECHO Checking Neo4j connection...
python %BIONN_DIR%\test_neo4j.py
if %ERRORLEVEL% NEQ 0 (
    ECHO Warning: Neo4j connection issues detected
)

:: Initialize Logic-LLM
ECHO Setting up Logic-LLM...
if exist "%LOGIC_LLM_DIR%\models\logic_program.py" (
    ECHO Logic-LLM models found
    set PYTHONPATH=%PYTHONPATH%;%LOGIC_LLM_DIR%;%LOGIC_LLM_DIR%\models
) else (
    ECHO Warning: Logic-LLM models not found at %LOGIC_LLM_DIR%\models
)

:: Start quantum bridge
ECHO Initializing Quantum Bridge...
python %BIONN_DIR%\test_quantum_stdp.py
if %ERRORLEVEL% NEQ 0 (
    ECHO Warning: STDP initialization issues detected
)

:: Test logic integration
ECHO Testing Logic Integration...
python %BIONN_DIR%\test_logic_integration.py

ECHO ===============================
ECHO System startup complete
ECHO - BioNN Directory: %BIONN_DIR%
ECHO - Logic-LLM Directory: %LOGIC_LLM_DIR%
ECHO Use 'run_batch.bat' to start processing

:: Keep window open
cmd /k