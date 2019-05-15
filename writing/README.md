# Compiling the Document

The actual thesis PDF is not under version control, and might only be uploaded
from time to time. Building the document from source is the suggested way to
get a copy of the work-in-progress.

## Requirements

- `texlive-full`: Not strictly neccesary to have the whole thing, but the packages used are too many to count.
- `pdflatex`: To compile the document
- `biber`: For citations
- `latexmk`: (Optional) Makes building significantly easier

## With `latexmk`

```bash
> # Compile document:
> make
> # Compile, and continue to watch for changes, recompiling as needed.
> make watch
```

## Without `latexmk`

```bash
> pdflatex Thesis.pdf
> biber Thesis
> pdflatex Thesis.pdf
> pdflatex Thesis.pdf
```
