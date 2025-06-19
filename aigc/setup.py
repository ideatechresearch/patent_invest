from setuptools import setup, find_packages
from config import Config

setup(
    name='aigc',
    version=Config.Version,
    packages=find_packages(),  # include=['agent']
    include_package_data=True,
    package_data={
        'aigc': [],  # 'data/*'
    },
)
# python setup.py sdist
# if __name__ == "__main__":
#     import subprocess, sys
#     cmd = [sys.executable, "setup.py", "sdist"]
#     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#     print(result.stdout)
#     # cmd = f"python3 setup.py sdist"
#     # p = subprocess.Popen(cmd, shell=True)
#     # p.wait()
