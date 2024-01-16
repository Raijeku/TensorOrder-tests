@echo off
for /f %%f in ('dir /b "%CD%"\CNF\') do (
    echo %%f >> result.txt
    docker run -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 < "%CD%\CNF\%%f" >> result.txt
    echo. >> result.txt
)
Rem docker run -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 < "benchmarks/3SAT/11_bit_0.cnf" >> test.txt
