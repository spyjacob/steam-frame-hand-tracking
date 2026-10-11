@echo off
rem Build driver_framehands.dll into framehands\bin\win64 (needs VS 2022 Build Tools).
setlocal
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /LD /MT /Iopenvr /Ithird_party\openvr_handsim src\driver_framehands.cpp third_party\openvr_handsim\hand_simulation.cpp /Fo:build\ /Fe:framehands\bin\win64\driver_framehands.dll /link ws2_32.lib || exit /b 1
echo built framehands\bin\win64\driver_framehands.dll
