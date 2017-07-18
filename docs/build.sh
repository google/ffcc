#!/bin/sh
bibtex fccc
pdflatex fccc
bibtex fccc_supp
pdflatex fccc_supp
pdfjam fccc.pdf fccc_supp.pdf -o merge.pdf
evince merge.pdf &
