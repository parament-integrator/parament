@echo off
SETLOCAL
set sourceFiles=deviceinfo.c diagonal_add.cu main.cpp mathhelper.cpp parament.cpp printFuncs.cpp debugfuncs.cpp control_expansion.cu
set nvccFlags=-lcublas
set outputDir=build

md build
cd src\cuda
@echo on
nvcc %nvccFlags% -DPARAMENT_LINK -o ..\..\%outputDir%\parament.exe %sourceFiles% || goto :error
nvcc %nvccFlags% -DPARAMENT_BUILD_DLL -o ..\..\%outputDir%\parament.dll --shared %sourceFiles% || goto :error
@echo off
cd ..\..
ENDLOCAL
goto :EOF


:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%