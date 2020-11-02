echo %path%
python -m venv parament_docs_py
call parament_docs_py\Scripts\activate.bat
pip install -r docs/requirements.txt
cd docs
make html
