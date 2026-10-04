@echo off
rem Installs the anime-typeset skill into Claude Code by mirroring this folder to
rem %USERPROFILE%\.claude\skills\anime-typeset (Claude Code only looks for skills there).
rem The master copy is this folder: edit here, then run install.bat.
set "SRC=%~dp0."
set "DST=%USERPROFILE%\.claude\skills\anime-typeset"
robocopy "%SRC%" "%DST%" /MIR /XD __pycache__ .git /XF install.bat README.md /NFL /NDL /NJH /NJS /NP >nul
if %ERRORLEVEL% GEQ 8 (
  echo Copy failed, robocopy code %ERRORLEVEL%
  exit /b 1
)
echo Skill installed: %DST%
exit /b 0
