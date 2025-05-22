"""
This `__init__.py` file makes the 'modules' directory a Python sub-package.
It also imports various sub-modules within the 'modules' directory to make them
accessible or, more importantly, to ensure that any components defined within
them (like specific analysis networks, synthesis networks, compressors, etc.)
are registered with the central `module_registry`.

Key modules imported include:
    - module_registry: The central registry for model components.
    - analysis: Contains analysis (encoder) network definitions.
    - synthesis: Contains synthesis (decoder) network definitions.
    - compressor: Defines various compression modules and entropy models.
    - hyper: Includes hyper-network architectures for hyperpriors.
    - autoregressive: Provides components for autoregressive entropy modeling.
    - stem: Defines input stem architectures.
    - task_predictors: Contains models for predicting task-related information.

By importing these, the registration decorators within each module (e.g.,
`@register_analysis_network`) are executed, populating the dictionaries in
`module_registry`. This allows for dynamic instantiation of these components
based on configuration files elsewhere in the project.
"""
# model/modules/__init__.py

# Import the registry itself first if other modules might access its dicts directly at import time
# (though usually they import functions like get_analysis_network)
from . import module_registry

# Import all modules that contain registered components
from . import analysis
from . import synthesis
from . import compressor
from . import hyper
from . import autoregressive
from . import stem
from . import task_predictors

# You might also want to import specific functions or classes to make them 
# available directly from model.modules, e.g.:
# from .module_registry import get_analysis_network, get_synthesis_network, ...
# from .stem import SharedInputStem
# from .task_predictors import TaskProbabilityModel