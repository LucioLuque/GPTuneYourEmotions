@echo off
REM -------------------------------------------------------
REM run_app.bat
REM Uso:
REM   .\run_app.bat flan
REM   .\run_app.bat gpt4o
REM -------------------------------------------------------

REM Si no se pasa argumento, asumimos "flan"
IF "%~1"=="" (
    SET MODE=flan
) ELSE (
    SET MODE=%~1
)

ECHO Modo seleccionado: %MODE%

IF /I "%MODE%"=="gpt4o" (
    REM -------------------
    REM MODO GPT-4o
    REM -------------------

    REM -------------------
    REM Importar variables de entorno definidas en credentials.bat
    REM Tiene que tener las siguientes lineas:
    REM SET AZURE_OPENAI_ENDPOINT=<tu_endpoint>
    REM SET AZURE_OPENAI_API_KEY=<tu_api_key>
    REM SET AZURE_OPENAI_DEPLOYMENT=<tu_deployment>
    REM -------------------
    
    SET BOT_BACKEND=gpt4o
    CALL credentials.bat

    ECHO Variables de entorno para GPT-4o cargadas desde credentials.bat.
) ELSE (
    REM -------------------
    REM MODO FLAN
    REM -------------------
    SET BOT_BACKEND=flan
    SET AZURE_OPENAI_ENDPOINT=
    SET AZURE_OPENAI_API_KEY=
    SET AZURE_OPENAI_DEPLOYMENT=

    ECHO Startup en modo FLAN local: variables limpiadas.
)

REM Finalmente, arrancamos la app
python app.py
