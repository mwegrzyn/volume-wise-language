#!/bin/bash
nb="$1"
outname="$2"

jupyter nbconvert ${nb} --to python --template pyExportTemplate.tpl
mv -f ${nb%.*}.py ${outname}
chmod a+x ${outname}
