# Environment for the TU Dresden HPC (Capella). Usage: source hpc_env.sh
export STABLEWM_HOME="/data/horse/ws/$USER-arc-lewm"  # data root on the horse workspace (datasets/, models/, evaluation/)
export UV_PROJECT_ENVIRONMENT="$STABLEWM_HOME/venv"  # uv installs the project venv here instead of .venv in the repo
export UV_CACHE_DIR="$STABLEWM_HOME/uv-cache"  # keep the uv download cache out of the home quota
export PATH="$HOME/.local/bin:$PATH"  # uv is installed in ~/.local/bin
export ACCOUNT="p_scads_lv_llm"  # Slurm project account
