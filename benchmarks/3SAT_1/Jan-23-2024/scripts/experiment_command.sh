#!/bin/bash
for filename in ../CNF/*.cnf; do
    { echo $(basename "${filepath%.*}") >> ../results/result.txt
    timeout 3600 docker run --net host --pid host --uts host --privileged --log-driver none -i tensororder:latest python /src/tensororder.py --planner="Factor-Flow" --weights="unweighted" --verbosity=5 < ../CNF/"$filename" >> "../results/result.txt"&
    echo >> "../results/result.txt" ; } &
done