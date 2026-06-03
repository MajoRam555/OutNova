@echo off
chcp 65001 >nul

echo ============================================================
echo  OutNova Indra Fusion — Iniciando Dashboard Analista
echo ============================================================

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado. Ejecuta setup.bat primero.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

cd backend
echo Iniciando Streamlit Dashboard en http://localhost:8501 ...
python -m streamlit run dashboard.py --server.port 8501 --server.address localhost

pause
