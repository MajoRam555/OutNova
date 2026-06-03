@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ============================================================
echo  OutNova Indra Fusion — Setup base
echo  Python 3.11 requerido
echo ============================================================
echo.

:: Verificar Python 3.11
python --version 2>&1 | findstr "3.11" >nul
if %errorlevel% neq 0 (
    echo [ADVERTENCIA] No se detectó Python 3.11 como versión activa.
    echo   Asegúrate de usar Python 3.11 para este proyecto.
    echo   Descarga: https://www.python.org/downloads/release/python-3119/
    echo.
    set /p CONTINUE="¿Continuar de todas formas? (s/n): "
    if /i "!CONTINUE!" neq "s" exit /b 1
)

:: Verificar/crear venv
if not exist "venv\" (
    echo [1/5] Creando entorno virtual...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: No se pudo crear el entorno virtual.
        exit /b 1
    )
) else (
    echo [1/5] Entorno virtual ya existe.
)

:: Activar venv
echo [2/5] Activando entorno virtual...
call venv\Scripts\activate.bat

:: Actualizar pip
echo [3/5] Actualizando pip...
python -m pip install --upgrade pip setuptools wheel --quiet

:: Instalar dependencias base
echo [4/5] Instalando dependencias base...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR al instalar requirements.txt
    exit /b 1
)
cd ..

:: Verificar ffmpeg
echo [5/5] Verificando FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] FFmpeg encontrado.
) else (
    echo.
    echo [ADVERTENCIA] FFmpeg NO encontrado en PATH.
    echo   FFmpeg es obligatorio para el pipeline de audio y video.
    echo   Descarga: https://ffmpeg.org/download.html
    echo   Agrega ffmpeg\bin a tu PATH del sistema.
    echo.
)

echo.
echo ============================================================
echo  Setup base completado.
echo.
echo  SIGUIENTE PASO OBLIGATORIO:
echo    Ejecuta: setup_ml.bat
echo    (instala PyTorch + Whisper para el pipeline biométrico)
echo.
echo  OPCIONAL (DeepFace):
echo    Ejecuta: setup_optional_deepface.bat
echo ============================================================
pause
