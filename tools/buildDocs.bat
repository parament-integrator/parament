echo %path%
python -m venv parament_docs_py
parament_docs_py\Scripts\activate.bat
pip install docs/requirements.txt
docs\make html
