@echo off
SETLOCAL
set sourceFiles=deviceinfo.c diagonal_add.cu main.cpp mathhelper.c parament.c printFuncs.c
set nvccFlags=-lcublas
set outputDir=build

md build
cd src\cuda2
@echo on
nvcc %nvccFlags% -o ..\..\%outputDir%\parament.exe %sourceFiles%
nvcc %nvccFlags% -o ..\..\%outputDir%\parament.dll --shared %sourceFiles%
@echo off
cd ..\..
ENDLOCAL