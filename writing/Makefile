# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: Thesis.pdf all clean

MAIN=Thesis
TEXFILES=Introduction.tex Appendix.tex

all: ${MAIN}.pdf

# Compile if needed.
${MAIN}.pdf: ${MAIN}.tex ${TEXFILES}
	latexmk -bibtex -f -pdf -pdflatex="pdflatex -synctex=1 -interaction=nonstopmode" $<

# Compile _and_ watch for future changes.
watch:
	latexmk -bibtex -pvc -f -pdf -pdflatex="pdflatex -synctex=1 -interaction=nonstopmode" ${MAIN}.tex

clean:
	latexmk -CA
	rm -f ${MAIN}.run.xml
