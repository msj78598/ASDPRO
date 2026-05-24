Set-Location -Path $PSScriptRoot
& "C:\Users\78598\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true *> streamlit.run.log
