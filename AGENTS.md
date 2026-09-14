# Research workflow

Keep derivations in notes/note.tex and reproducible checks in src/ and tests/.
Use Japanese prose, define each symbol and concept on first use, and do not assume access to chat history. Use standard terminology and numbered equations. Distinguish probability sample space, state-parameter space, and bulk spacetime.

Read the cited version of a primary paper before importing a formula. Check limiting cases and low-order expansions independently. Separate input assumptions, derived identities, numerical evidence, and unverified holographic interpretations. Do not claim novelty from a log-cosh fit alone.

Before updating results, run:

    python -m unittest discover -s tests -v
    OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
    latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build notes/note.tex

Inspect rendered PDF pages for overflow and unresolved references. GitHub Actions publishes the compiled PDF at docs/note.pdf and stores run artifacts. Also provide a compiled PDF in the conversation when a research note is updated.
