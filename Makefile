LATEXMK ?= latexmk
LUALATEX_FLAGS ?= -cd -lualatex -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all clean

all: paper.pdf qc_paper.pdf ajp_paper.pdf ajp_paper_v2.pdf ajp_cover.pdf speculative_paper.pdf

%.pdf: %.tex
	$(LATEXMK) $(LUALATEX_FLAGS) $<

clean:
	$(LATEXMK) -C paper.tex
	$(LATEXMK) -C qc_paper.tex
	$(LATEXMK) -C speculative_paper.tex
	$(LATEXMK) -C ajp_paper.tex
	$(LATEXMK) -C ajp_paper_v2.tex
	$(LATEXMK) -C ajp_cover.tex
