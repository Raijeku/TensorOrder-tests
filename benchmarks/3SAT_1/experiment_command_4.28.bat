@echo off
for /f %%f in ('dir /b "%CD%"\CNF_4.28\') do (
    echo %%f
    echo %%f >> result_CNF_4.28.txt
    docker run -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 < "%CD%\CNF_4.28\%%f" >> result_CNF_4.28.txt
    echo. >> result_CNF_4.28.txt
)
