@echo off
chcp 65001 >nul

echo ============================================================
echo  OutNova Indra Fusion — Setup ML (PyTorch + Whisper)
echo  ADVERTENCIA: Puede descargar varios GB
echo ============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado. Ejecuta setup.bat primero.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Instalando PyTorch CPU...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Instalando Whisper y dependencias ML...
cd backend
pip install -r requirements-ml.txt
if %errorlevel% neq 0 (
    echo ERROR al instalar requirements-ml.txt
    pause
    exit /b 1
)
cd ..

echo.
echo ============================================================
echo  Setup ML completado.
echo  PyTorch y Whisper instalados.
echo.
echo  Para iniciar el servidor: start_server.bat
echo ============================================================
pause
