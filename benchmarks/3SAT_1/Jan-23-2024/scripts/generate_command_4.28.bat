SETLOCAL EnableDelayedExpansion

@echo off
for /L %%N in (10, 1, 30) do (
    set /a number = %%N * 428
    set /a clauses = !number:~0,-2!
    set /a instances = 310 - %%N*10 
    echo %%N variables and !clauses! clauses
    echo !instances! instances
    for /L %%G in (1, 1, !instances!) do (
        cnfgen --output ../CNF/%%N_bit_!clauses!_%%G.cnf randkcnf 3 %%N !clauses!
    )
)