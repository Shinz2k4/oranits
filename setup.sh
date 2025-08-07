#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
#!/bin/bash

# Kích hoạt môi trường ảo Python phù hợp với từng hệ điều hành
if [[ -f "venv/bin/activate" ]]; then
    # Linux hoặc macOS
    echo "Activating virtual environment (Linux/macOS)..."
    source venv/bin/activate
elif [[ -f "venv/Scripts/activate" ]]; then
    # Windows (Git Bash)
    echo "Activating virtual environment (Windows Git Bash)..."
    ./venv/Scripts/activate
else
    echo "Could not find virtual environment activation script."
    echo "Please ensure the venv is created."
    exit 1
fi



echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo " Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo " No requirements.txt found. Skipping dependency installation."
fi

if [ ! -d "task" ]; then
    echo "Creating 'task' directory..."
    mkdir task
else
    echo "'task' directory already exists."
fi

echo "Running ITS_based.py..."
python ./src/physic_definition/system_base/ITS_based.py
