LATEXMK ?= latexmk
LUALATEX_FLAGS ?= -cd -lualatex -interaction=nonstopmode -halt-on-error -file-line-error

PAPER_TEX := paper.tex

.PHONY: all paper clean

all: paper

paper: paper.pdf

%.pdf: %.tex
	$(LATEXMK) $(LUALATEX_FLAGS) $<

clean:
	$(LATEXMK) -C $(PAPER_TEX)