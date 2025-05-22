"""
This `__init__.py` file makes the 'layers' directory a Python sub-package,
intended to contain various neural network layer definitions.

While currently empty, it would typically be used to import specific layer
classes from the modules within this directory (e.g., `conv.py`, `transf.py`,
`film.py`, `recon.py`, `composed.py`, `preconfigured.py`) to make them
directly accessible via `model.modules.layers.<LayerName>` if desired,
or to ensure any layer-specific registries are handled.
"""
