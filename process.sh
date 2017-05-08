#!/bin/sh

f=$1

echo $f; grep Successfully $f | sed 's/Successfully remembered: \([0-9]*\) out of 10/\1/' > `basename $f .log`.data;
