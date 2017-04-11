#!/bin/bash

# get final found length of path
function length
{
    printf "%s" $(cat $1 | \
                  grep "New global minimum" | \
                  tail -n1 | \
                  cut -d">" -f2 | \
                  cut -d" " -f3)
}

# get time
function path_time
{
    printf "%s" $(tail -n1 $1 | \
                  cut -d" " -f 3)
}

# get accuracy
function acc
{
    printf "%s" $(cat $1 | \
                  grep "New global minimum" | \
                  tail -n1 | \
                  cut -d">" -f2 | \
                  cut -d" " -f5)
}

out=${1:-data.csv}

printf "1,2,4,8,16,32,64,gpu,\n" > $out

let count=0
for f in $(ls results/* | sort -V); do
    printf "%s," $(path_time $f) >> $out
    let count=$count+1

    if [ $((count%8)) -eq 0 ]; then
        printf "\n" >> $out
    fi
done


# reserved data
# printf "%s,"  $(length $f)
# printf "%s,"  $(acc $f) | sed "s/%//"
