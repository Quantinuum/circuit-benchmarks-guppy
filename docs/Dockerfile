##### LOCAL IMAGE
# This stage will require mounting the local directory during `docker run`
# By not copying local data into the docker image, the image can be
# cached and building documentation will be much faster

# Get the image hosted by astral.sh that already has uv installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS mount-local

# Set environment variables
# Noninteractive frontend prevents tons of CLI commands from requesting feedback
ENV DEBIAN_FRONTEND=noninteractive \
    # Don't cache packages. The image is ephemeral, so caching is just wasting space
    PIP_NO_CACHE_DIR=1 \
    # PlantUML variables
    PLANTUML_VERSION=1.2025.3 \
    PLANTUML_JAR=/opt/plantuml/plantuml.jar \
    # Set the path so that we don't need to use "uv run". This ends up being a lot faster
    PATH="/app/.venv/bin:$PATH" \
    # Set the python path to point to the sourcecode location
    PYTHONPATH="/app/src"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    # Get java and graphviz to run plantUML
    openjdk-17-jre-headless \
    graphviz \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install PlantUML jar manually
RUN mkdir -p /opt/plantuml \
    && curl -L -o ${PLANTUML_JAR} "https://github.com/plantuml/plantuml/releases/download/v${PLANTUML_VERSION}/plantuml-${PLANTUML_VERSION}.jar"

# Set workdir inside the container
WORKDIR /app

# Copy only files needed to install dependencies
COPY pyproject.toml uv.lock README.md ./

# Don't install the dev and test groups
RUN uv sync --no-default-groups && \
    # Install locally, otherwise autodoc can't find the package
    uv pip install -e . && \
    # Install the docs group
    uv pip install --group docs

# Define the entrypoint. This forces the container to stay alive after "docker start"
# This is important because it allows "docker exec" to be issued afterwards, which
# separates the logic of creating and mounting a docker image from executing commands
# inside a docker image, and allows the same execution logic to be used when
# running locally or in CI
CMD ["tail", "-f", "/dev/null"]



##### CI/CD IMAGE
# Copy all data into the image for CI/CD pipelines
# This helps ensure maximum robustness, but requires an image rebuild every
# time data in the repo changes
FROM mount-local AS ci

# Copy necessary files in
COPY src/ /app/src/
COPY notebooks/ /app/notebooks/
COPY docs/ /app/docs/
COPY README.md /app/README.md
