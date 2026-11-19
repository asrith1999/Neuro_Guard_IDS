@echo off
setlocal
cd /d "%~dp0"
echo.
echo ====================================
echo  NeuroGuard IDS - Beginner Demo
echo ====================================
echo.
if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python virtual environment...
  py -3.12 -m venv .venv 2>nul
  if errorlevel 1 py -3.11 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Python not found. Install Python 3.12 and enable the Python launcher.
  pause
  exit /b 1
)
echo Installing required Python packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
if not exist "data\synthetic_training.csv" (
  echo Creating fictional demo dataset...
  ".venv\Scripts\python.exe" generate_data.py
  if errorlevel 1 goto error
)
if not exist "models\model.joblib" (
  echo Training the neural network...
  ".venv\Scripts\python.exe" train_model.py
  if errorlevel 1 goto error
)
echo Starting local dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
goto end
:error
echo.
echo Something failed. Please copy the error displayed above.
pause
exit /b 1
:end
endlocal
