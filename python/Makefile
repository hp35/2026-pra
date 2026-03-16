#
# Makefile designed for use with gcc
#
# Copyright (C) 2026 under GPLv3, Fredrik Jonsson
#

#
# TeX-specific / document preparing declarations:
#
TEX       = tex
DVIPS     = dvips
DVIPSOPTS = -D1200 -E
METAPOST  = mpost
PS2PDF    = ps2pdf
PDFCROP   = pdfcrop
PDF2SVG   = pdf2svg

#
# The POINCARE program for mapping paths of Stokes parameters onto the
# Poincare sphere.
#
POINCAREPATH = ../../2025-pulsedopa/poincare
POINCARE = poincare

all:
	make poincaremap-pump
	make poincaremap-signal

poincaremap-pump:
	make -C $(POINCAREPATH)
	@$(POINCAREPATH)/$(POINCARE) --verbose --normalize \
		--draw_hidden_dashed \
		--inputfile graph-06-stokes-pump.dat \
		--outputfile graph-06-stokes-pump.mp \
		--axislengths 0.3 3.5 0.3 1.45 0.3 1.45 \
		--axislabels  "S_1(s)/S_0(s)" bot \
				"S_2(s)/S_0(s)" bot \
				"S_3(s)/S_0(s)" rt \
		--rotatephi 15.0 --rotatepsi -15.0 --shading 0.75 0.99 \
		--rhodivisor 50  --phidivisor 80  --scalefactor 26.0 \
		--draw_paths_as_arrows \
		--paththickness 0.8 --arrowthickness 0.4
	@$(METAPOST) graph-06-stokes-pump.mp
	@$(TEX) -jobname=graph-06-stokes-pump "\input epsf.tex\nopagenumbers\
	  \centerline{\epsfxsize=155mm\epsfbox{graph-06-stokes-pump.1}}\bye"
	@$(DVIPS) $(DVIPSOPTS) graph-06-stokes-pump -o graph-06-stokes-pump.eps
	@$(PS2PDF) graph-06-stokes-pump.eps graph-06-stokes-pump.pdf
	@$(PDFCROP) graph-06-stokes-pump.pdf graph-06-stokes-pump-crop.pdf
	@$(PDF2SVG) graph-06-stokes-pump-crop.pdf graph-06-stokes-pump.svg

poincaremap-signal:
	make -C $(POINCAREPATH)
	@$(POINCAREPATH)/$(POINCARE) --verbose --normalize \
		--draw_hidden_dashed \
		--inputfile graph-06-stokes-signal.dat \
		--outputfile graph-06-stokes-signal.mp \
		--axislengths 0.3 3.5 0.3 1.45 0.3 1.45 \
		--axislabels  "S_1(s)/S_0(s)" bot \
				"S_2(s)/S_0(s)" bot \
				"S_3(s)/S_0(s)" rt \
		--rotatephi 15.0 --rotatepsi -15.0 --shading 0.75 0.99 \
		--rhodivisor 50  --phidivisor 80  --scalefactor 26.0 \
		--draw_paths_as_arrows \
		--paththickness 0.8 --arrowthickness 0.4
	@$(METAPOST) graph-06-stokes-signal.mp
	@$(TEX) -jobname=graph-06-stokes-signal "\input epsf.tex\nopagenumbers\
	  \centerline{\epsfxsize=155mm\epsfbox{graph-06-stokes-signal.1}}\bye"
	@$(DVIPS) $(DVIPSOPTS) graph-06-stokes-signal -o graph-06-stokes-signal.eps
	@$(PS2PDF) graph-06-stokes-signal.eps graph-06-stokes-signal.pdf
	@$(PDFCROP) graph-06-stokes-signal.pdf graph-06-stokes-signal-crop.pdf
	@$(PDF2SVG) graph-06-stokes-signal-crop.pdf graph-06-stokes-signal.svg

#		--auxsource stokepathlabels.mp

clean:
	-rm -Rf *~ *.dat *.dvi *.log
