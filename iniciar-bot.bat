@echo off
chcp 65001 >nul
title Medley Bot
rem Liga o bot do Telegram. De dois cliques neste arquivo (ou num atalho para ele).
rem Para parar: feche esta janela ou aperte Ctrl+C.

rem Entra na pasta onde este arquivo esta, de onde quer que ele seja aberto
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo Nao encontrei o ambiente do projeto ^(.venv^).
    echo Rode primeiro, nesta pasta:  python -m venv .venv  e depois  .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
if not exist ".env" (
    echo Nao encontrei o arquivo .env com as chaves. Veja o README.
    pause
    exit /b 1
)

:inicio
echo.
echo ===== Ligando o bot (%date% %time:~0,8%) =====
.venv\Scripts\python -m medley.bot
if %errorlevel%==0 goto fim
echo.
echo O bot parou com erro. Ligando de novo em 15 segundos...
echo ^(feche esta janela para cancelar^)
timeout /t 15 /nobreak >nul
goto inicio

:fim
echo.
echo O bot foi desligado.
pause
