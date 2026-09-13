"""pmkvq_cumulative_error: canonical modules for H1.

The falsifiable growth-law dissection of the PM-KVQ 'cumulative error' claim.
Kept as a package (rather than repo-root modules) so imports are explicit:

    from pmkvq import quantizer, observables, analysis, style
    from pmkvq import cache_hook, run_experiment

Entrypoints (scripts/, tests/, the notebook) put the repo root on sys.path so
`pmkvq` resolves without an editable install. `style`, `cache_hook`, and
`run_experiment` pull heavier dependencies only when their submodule is
imported, so importing the package itself stays light.
"""
