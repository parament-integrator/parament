echo %path%
python -m venv parament_py
call parament_py\Scripts\activate.bat
pip install -r ./src/python/requirements.txt
cd src
python setup.py bdist_wheel
python setup.py sdist
