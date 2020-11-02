echo %path%
python -m venv parament_py
call parament_py\Scripts\activate.bat
pip install -r ./src/python/requirements.txt
pip install src\dist\parament-0.1-py3-none-win_amd64.whl
pytest --cov --pyargs parament
