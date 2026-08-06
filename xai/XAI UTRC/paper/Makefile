# Build the URTC paper. Run from this directory: `make`
#
# The four-step sequence is not optional: bibtex needs the .aux written by the
# first pdflatex pass, and the two trailing passes resolve \cite keys and settle
# float placement. Stopping after one pass yields 20 undefined citations, an
# empty References section, and a short paper.

.PHONY: all clean check

all: main.pdf

main.pdf: main.tex ../references.bib $(wildcard ../figures/*.pdf)
	pdflatex -interaction=nonstopmode main.tex
	bibtex   main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	@$(MAKE) --no-print-directory check

# Fail loudly on the failure modes that produce a plausible-looking but wrong PDF.
check:
	@pages=$$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $$2}'); \
	undef=$$(grep -c 'Citation .* undefined' main.log 2>/dev/null | tail -1); \
	miss=$$(grep -c 'not found' main.log 2>/dev/null | tail -1); \
	: $${undef:=0}; : $${miss:=0}; \
	echo "pages=$$pages undefined_citations=$$undef missing_files=$$miss"; \
	if [ "$$pages" != "5" ]; then \
	  echo "WARNING: expected 5 pages. 4 usually means main.tex was compiled without"; \
	  echo "         ../figures and ../references.bib; 6 usually means a XeTeX engine."; \
	fi; \
	if [ "$$undef" != "0" ]; then echo "ERROR: undefined citations - bibtex did not run"; exit 1; fi; \
	if [ "$$miss" != "0" ]; then echo "ERROR: missing figures - check ../figures/"; exit 1; fi

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.synctex.gz
