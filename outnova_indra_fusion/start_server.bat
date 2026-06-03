@echo off
chcp 65001 >nul

echo ============================================================
echo  OutNova Indra Fusion — Iniciando servidor
echo ============================================================

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado. Ejecuta setup.bat primero.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

cd backend
echo Iniciando FastAPI en http://localhost:8000 ...
python start_server.py

pause
