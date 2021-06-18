echo %path%
python -m venv parament_py
call parament_py\Scripts\activate.bat
pip install -r ./src/python/requirements.txt || goto :error
cd src
python setup.py bdist_wheel || goto :error
python setup.py sdist || goto :error
deactivate
echo Building pyparament successful. Artifacts are located in `src\dist`.
goto :EOF


:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%