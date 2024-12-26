@echo off
ECHO Setting up Scale-Agnostic ML Environment...

:: Initialize Mambaforge if available
IF EXIST "%UserProfile%\mambaforge\Scripts\activate.bat" (
    CALL %UserProfile%\mambaforge\Scripts\activate.bat
) ELSE (
    IF EXIST "%UserProfile%\miniconda3\Scripts\activate.bat" (
        CALL %UserProfile%\miniconda3\Scripts\activate.bat
    ) ELSE (
        ECHO Error: Neither Mambaforge nor Miniconda found.
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
