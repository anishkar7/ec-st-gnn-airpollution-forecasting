@echo off
echo =======================================================
echo  Starting EC-STGNN Air Pollution Digital Twin Server   
echo =======================================================
pip install -r ../Support/requirements.txt
streamlit run main.py
pause