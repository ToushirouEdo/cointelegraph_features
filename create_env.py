import yaml
import subprocess
import sys

with open('my_env.yml', 'r') as file:
    config = yaml.safe_load(file)

dependencies = config.get('dependencies', [])

for dependency in dependencies:
    if isinstance(dependency, dict):
        # Handle pip-specific dependencies
        if 'pip' in dependency:
            for pip_package in dependency['pip']:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_package])
    else:
        # Handle conda-like dependencies (you might need to adjust this)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])



