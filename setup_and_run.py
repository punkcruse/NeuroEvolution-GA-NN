
import os
import subprocess
import platform

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

venv_dir = "venv"
if not os.path.exists(venv_dir):
    run_command("python -m venv venv")

pip_path = os.path.join(venv_dir, "Scripts", "pip.exe") if platform.system() == "Windows" else os.path.join(venv_dir, "bin", "pip")
python_path = os.path.join(venv_dir, "Scripts", "python.exe") if platform.system() == "Windows" else os.path.join(venv_dir, "bin", "python")

run_command(f"{pip_path} install torch torchvision scikit-learn matplotlib pandas")
run_command(f"{python_path} main.py")
