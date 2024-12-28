@echo off
ECHO Setting up Scale-Agnostic ML Environment...

:: Initialize miniconda if available
IF EXIST "F:\miniconda3\Scripts\activate.bat" (
    CALL F:\miniconda3\Scripts\activate.bat)

     ELSE (
    IF EXIST "%UserProfile%\Scripts\activate.bat" (
        CALL F:\miniconda3\Scripts\activate.bat
    ) ELSE (
        ECHO Error: Miniconda not found.
        EXIT /B 1
    )
)

:: Create and activate environment
CALL conda create -n scale_agnostic python=3.10 -y
CALL conda activate scale_agnostic

:: Install PyTorch with CUDA support
CALL conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

:: Install other dependencies
CALL pip install -r ..\scale_agnostic_unconditional_generation\requirements.txt

:: Create symbolic link to main ML lab
mklink /D "%~dp0\scale_agnostic" "..\scale_agnostic_unconditional_generation"

ECHO Environment setup complete!
