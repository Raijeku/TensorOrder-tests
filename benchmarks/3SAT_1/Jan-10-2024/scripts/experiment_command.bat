@echo off
for /f %%f in ('dir /b "%CD%"\..\CNF\') do (
    echo %%f >> "..\results\result.txt"
    wsl.exe timeout 120 docker run --net None --pid host --uts host --privileged --log-driver none -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 --timeout=20 < "%CD%\..\CNF\%%f" >> "..\results\result.txt"
    echo. >> "..\results\result.txt"
)
Rem docker run -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 < "benchmarks/3SAT/11_bit_0.cnf" >> test.txt
