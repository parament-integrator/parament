echo %path%
python -m venv parament_py
call parament_py\Scripts\activate.bat
pip install -r ./src/python/requirements.txt  || goto :error
pip install src\dist\parament-0.1-py3-none-win_amd64.whl || goto :error
pytest --cov=parament --cov-report xml --cov-report term --junitxml=report_test.xml --pyargs parament || goto :error
deactivate
cd ..
goto :EOF


:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%