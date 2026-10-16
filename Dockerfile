# Telecom 5G Outage Prediction -- container image.
#
# Packages the full pipeline (decision_agent, orchestrator_v3/llm, RAG,
# Streamlit dashboard, Query Agent) so it runs the same way regardless of
# host machine. Does NOT bundle trained models or data -- those are
# gitignored and machine-specific; mount them at runtime (see
# docker-compose.yml) rather than baking them into the image, so retraining
# or swapping datasets doesn't require a rebuild.
#
# BUILD (from the project root, D:\telecom_project_v2\telecom_sim_v2):
#     docker build -t telecom-outage-prediction .
#
# RUN (single command, CLI):
#     docker run --rm -v %cd%\data:/app/data -v %cd%\agents\models:/app/agents/models ^
#         telecom-outage-prediction python -m agents.orchestrator_v3 --tower T0000
#
# RUN (Streamlit dashboard, port 8501):
#     docker run --rm -p 8501:8501 -v %cd%\data:/app/data -v %cd%\agents\models:/app/agents/models ^
#         telecom-outage-prediction streamlit run agents/dashboard.py --server.address=0.0.0.0
#
# Prefer docker-compose.yml for either of the above -- it wires the mounts
# and ports for you, see that file's comments.

FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for lightgbm/xgboost wheels that need compiling
# on some platforms; curl only if you later add a healthcheck script.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual project code. data/ and agents/models/ are intentionally
# NOT copied here (see .dockerignore) -- they're mounted as volumes instead.
COPY . .

# Ollama, if you use --provider ollama, runs as a SEPARATE container/service
# (see docker-compose.yml) -- this image only contains the Python pipeline,
# it does not bundle Ollama or any model weights.

EXPOSE 8501

# Default command: drop into the CLI orchestrator help text. Override this
# at `docker run` time (see examples above) for the dashboard or a specific
# tower analysis -- there's no single "right" default command for a project
# with this many entry points (CLI x2, dashboard, query agent, eval harness).
CMD ["python", "-m", "agents.orchestrator_v3", "--help"]
