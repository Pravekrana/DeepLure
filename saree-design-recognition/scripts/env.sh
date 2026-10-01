# Source from the project root. All caches stay in project-created private storage.
export TORCH_HOME="$PWD/private/torch-cache"
export MPLCONFIGDIR="$PWD/private/matplotlib-cache"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=2
export JUPYTER_RUNTIME_DIR="$PWD/private/jupyter-runtime"
export IPYTHONDIR="$PWD/private/ipython"
