# Building the paper

`main.tex` is **not self-contained**. It reads two things from the parent
directory:

```
xai/XAI URTC v2/
├── references.bib        <- \bibliography{../references}
├── figures/              <- \graphicspath{{../figures/}}
│   ├── fig2_global.pdf
│   ├── fig3_dependence.pdf
│   ├── fig4_event_maps.pdf
│   └── fig5_cases.pdf
└── paper/
    ├── main.tex          <- compile from HERE
    └── main.pdf
```

Copying `main.tex` somewhere on its own and compiling it produces **4 pages with
missing figures and no reference list**, because all four `\includegraphics`
calls and the bibliography silently fail. That is the single most common way to
get a wrong-looking build.

## Correct build

From inside `paper/`:

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Or just:

```bash
make          # from paper/
```

**All four passes are required.** `bibtex` needs the `.aux` file that the first
`pdflatex` writes, and the two later passes are what resolve `\cite` keys and
settle float placement. A single pass leaves 20 undefined citations and an empty
References section, which is why the page count comes out short.

## Expected output

- **5 pages**, US Letter (612 x 792 pt)
- 0 errors, 0 undefined citations
- Table I on p2; Fig. 1 p3; Fig. 2 p4; Figs. 3-4 p5

If you get 4 pages, you are compiling `main.tex` in isolation. If you get 6, you
are using `tectonic` or another XeTeX-based engine, whose font metrics differ
from pdfLaTeX. **Use pdfLaTeX for the submission copy**: the 5-page limit is
measured against it.

## Overleaf

Upload the whole `XAI URTC v2/` directory, not just `paper/`. Set the main
document to `paper/main.tex` and the compiler to **pdfLaTeX** (Menu -> Compiler).
Overleaf runs bibtex automatically.

## Requirements

`IEEEtran.cls` plus `graphicx`, `amsmath`, `booktabs`, `array`,
`hyperref`, `balance`, `stfloats`. All are in a standard TeX Live install
(`brew install texlive`, or TeX Live / MiKTeX on Linux and Windows).
