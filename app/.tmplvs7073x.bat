@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "D:\anaconda\condabin\conda.bat" activate "e:\git_demo\app"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@e:\git_demo\app\python.exe -Wi -m compileall -q -l -i C:\Users\LOGESH\AppData\Local\Temp\tmp_x1rc6rr -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
