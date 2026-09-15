# Research workflow

Keep self-contained Japanese derivations in notes/ and reproducible checks in src/ and tests/. Preserve earlier notes when adding an independent supplement. Define each symbol and concept on first use and do not assume access to chat history. Use standard terminology and numbered equations. Distinguish probability sample space, state-parameter space, and bulk spacetime.

Read the cited version of a primary paper before importing a formula. Check limiting cases and low-order expansions independently. Separate input assumptions, derived identities, numerical evidence, and unverified holographic interpretations. Do not claim novelty from a log-cosh fit or a standard amplitude-phase decomposition alone.

Before updating results, run:

    python -m unittest discover -s tests -v
    OPENBLAS_NUM_THREADS=1 python src/krylov.py --output results
    OPENBLAS_NUM_THREADS=1 python src/geodesic.py --output results/geodesic
    OPENBLAS_NUM_THREADS=1 python src/state_distance.py --output results/state_distance
    OPENBLAS_NUM_THREADS=1 python src/phase_space.py --output results/phase_space
    for topic in note geodesic state_distance phase_space; do
        latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=build "notes/${topic}.tex"
    done

Inspect rendered PDF pages for overflow and unresolved references. GitHub Actions publishes compiled PDFs under docs/ and stores run artifacts. Also provide a compiled PDF in the conversation when a research note is updated. Never distribute font files.

The phase_space module works in the fixed orthonormal chord basis. The quantity arg(<A>) is an observable coherence phase used for semiclassical comparison, not the expectation of a globally assumed momentum operator. Pure-state reconstruction from adjacent complex coherences requires a connected nonzero support; P and J alone have a sine-branch ambiguity. State restrictions must be stated explicitly.
