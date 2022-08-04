#!/bin/bash
sudo apt install -y python3-pip
sudo apt install -y python3.8-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
deactivate
mkdir -p DATASET/
mkdir -p resources/