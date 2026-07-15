@echo off
SETLOCAL

chcp 65001 >nul
set "PYTHONUTF8=1"
V:
cd "V:\PROGRAMING\Scratch-Scripts\AI_FaceMat"

set "UV_CACHE_BASE=V:\PROGRAMING\Scratch-Scripts\AI_FaceMat"
if exist "G:\" set "UV_CACHE_BASE=G:\Scratch-AI-matte-cache"

if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"
if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"
set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"
set "TMP=%UV_CACHE_BASE%\.uv-tmp"
set "TEMP=%UV_CACHE_BASE%\.uv-tmp"
set "UV_LINK_MODE=copy"

set "SCRATCH_PORT=8080"
set "SCRATCH_API_KEY="

"C:\Users\oscar\.local\bin\uv.exe" run ^
	--verbose ^
	--default-index https://pypi.org/simple ^
	--index https://download.pytorch.org/whl/cu124 ^
	--index-strategy unsafe-best-match ^
	AI_FaceMat.py %*
pause

ENDLOCAL