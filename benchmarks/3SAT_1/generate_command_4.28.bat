SETLOCAL EnableDelayedExpansion

@echo off
for /L %%N in (4, 1, 27) do (
    set /a number = %%N * 428
    set /a clauses = !number:~0,-2!
    echo %%N variables and !clauses! clauses
    for /L %%G in (1, 1, 1000) do (
        cnfgen --output CNF_4.28/%%N_bit_!clauses!_%%G.cnf randkcnf 3 %%N !clauses!
    )
)