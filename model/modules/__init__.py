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
from .backbones import MultiTailResNetBackbone

# You might also want to import specific functions or classes to make them 
# available directly from model.modules, e.g.:
# from .module_registry import get_analysis_network, get_synthesis_network, ...
# from .stem import SharedInputStem
# from .task_predictors import TaskProbabilityModel