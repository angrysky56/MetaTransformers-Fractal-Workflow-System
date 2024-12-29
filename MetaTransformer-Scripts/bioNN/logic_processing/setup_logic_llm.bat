@echo off
ECHO Setting up Logic-LLM Integration...
ECHO ===============================

:: Set paths
set BASE_DIR=%~dp0
set LOGIC_LLM_DIR=%BASE_DIR%Logic-LLM

:: Check environment activation
IF "%CONDA_DEFAULT_ENV%" == "metatransformers" (
    ECHO metatransformers environment already active
) ELSE (
    CALL conda activate metatransformers
)

:: Initialize Logic-LLM
ECHO Initializing Logic-LLM...
cd %LOGIC_LLM_DIR%

:: Run test program generation
ECHO Testing logic program generation...
python models/logic_program.py ^
    --api_key "%OPENAI_API_KEY%" ^
    --dataset_name "test" ^
    --split dev ^
    --model_name "gpt-4" ^
    --max_new_tokens 1024

:: Run test inference
ECHO Testing logic inference...
python models/logic_inference.py ^
    --model_name "gpt-4" ^
    --dataset_name "test" ^
    --split dev ^
    --backup_strategy "random"

ECHO ===============================
ECHO Logic-LLM setup complete
ECHO Run specific Logic-LLM commands from %LOGIC_LLM_DIR%

:: Return to original directory
cd %BASE_DIR%

cmd /k