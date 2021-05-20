@echo off
REM Copyright 2021 Konstantin Herb, Pol Welter. All Rights Reserved.
REM 
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM     http://www.apache.org/licenses/LICENSE-2.0
REM 
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM =========================================================================

SETLOCAL
set sourceFiles=deviceinfo.c diagonal_add.cu main.cpp mathhelper.cpp parament.cpp printFuncs.cpp debugfuncs.cpp control_expansion.cu
set nvccFlags=-lcublas
set outputDir=build

md build
cd src\cuda
@echo on
nvcc %nvccFlags% -DNDEBUG -DPARAMENT_LINK -o ..\..\%outputDir%\parament.exe %sourceFiles% || goto :error
nvcc %nvccFlags% -DNDEBUG -DPARAMENT_BUILD_DLL -o ..\..\%outputDir%\parament.dll --shared %sourceFiles% || goto :error
@echo off
cd ..\..
ENDLOCAL
goto :EOF


:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%