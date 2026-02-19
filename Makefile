#
# Makefile for compilation of reports written in plain TeX.
#
# Copyright (C), Fredrik Jonsson, under GPL 3.0. See enclosed LICENSE.
#
PROJECT = phaseevo

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).ps
	ps2pdf $(PROJECT).ps $(PROJECT).pdf

$(PROJECT).ps: $(PROJECT).dvi
	dvips -D1200 -ta4 $(PROJECT).dvi -o $(PROJECT).ps

$(PROJECT).dvi: $(PROJECT).tex
	tex $(PROJECT).tex
	tex $(PROJECT).tex

archive:
	make -ik clean
	tar --directory=../ -cf ../$(PROJECT).tar $(PROJECT)

clean:
	rm -Rf *~ *.aux *.toc *.log *.dvi *.ps *.tar.gz
