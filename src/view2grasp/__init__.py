"""View2Grasp: vision-grounded, observation-only manipulation."""
import os
# Robot IK consists of many tiny linear algebra calls: avoid BLAS oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
__version__ = "0.1.0"
