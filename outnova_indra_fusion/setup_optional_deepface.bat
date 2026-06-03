@echo off
chcp 65001 >nul

echo ============================================================
echo  OutNova Indra Fusion — Setup DeepFace (OPCIONAL)
echo.
echo  ADVERTENCIA:
echo    - Puede tardar 10-30 minutos
echo    - Descarga varios GB (TensorFlow, modelos)
echo    - Puede generar conflictos de dependencias
echo    - El sistema funciona sin DeepFace
echo ============================================================
echo.
set /p CONFIRM="¿Continuar instalación de DeepFace? (s/n): "
if /i "%CONFIRM%" neq "s" (
    echo Instalación cancelada.
    pause
    exit /b 0
)

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado. Ejecuta setup.bat primero.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Instalando DeepFace y TensorFlow...
cd backend
pip install -r requirements-optional.txt
if %errorlevel% neq 0 (
    echo.
    echo [ADVERTENCIA] Hubo errores durante la instalación.
    echo   DeepFace puede no funcionar correctamente.
    echo   El sistema base seguirá funcionando sin DeepFace.
)
cd ..

echo.
echo Para activar DeepFace en el pipeline:
echo   Agrega USE_DEEPFACE=1 en tu archivo .env o como variable de entorno.
echo.
pause
