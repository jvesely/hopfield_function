#!/bin/bash

LOC=`dirname $(readlink -f "$0")`
if [ -f $LOC/translate.sh ] ; then
	source $LOC/translate.sh
fi

echo -n '
set terminal eps noenhanced
set output "plot.eps"

set style boxplot
set style data boxplot

set yrange [0:10]
set xrange [0:'"(($# + 1))"']

set xtics ("" 0'
for i in `seq $#`; do
	T=`translate "${!i}"`
	echo -n ", \"$T\" $i"
done
echo -n ')

plot \
'
for i in `seq $#`; do
	echo "'${!i}' using($i):1 notitle, \\"
done
