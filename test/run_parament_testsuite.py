import subprocess, os, shlex

# this line will be replaced before the script is deployed on Kaggle
GITHUB_SHA = "{{ GITHUB_SHA }}"
GITHUB_REPOSITORY = "{{ GITHUB_REPOSITORY }}"

print('=== Installing parament...')
my_env = os.environ.copy()
my_env["NVCC_ARGS"] = "-arch=compute_37"
cmd = f'pip install pytest-cov git+https://github.com/{GITHUB_REPOSITORY}@{GITHUB_SHA}#subdirectory=src'
print('>', cmd)
args = shlex.split(cmd)
process = subprocess.run(args,
                         stdout=subprocess.PIPE,
                         universal_newlines=True)
process.check_returncode()

print('=== Running test suite...')
import pytest
assert 0 == pytest.main(["--cov=parament", "--cov-report", "xml", "--cov-report", "term", "--junitxml=report_test.xml",
                         "--pyargs", "parament"]),\
    "Test suite failed"
