default:
  just --list



################################################################################
## Workflow
################################################################################

[group("workflow")]
venv:
    just clean-venv
    uv venv
    uv lock
    uv sync
    #uv run pre-commit install


[group("clean")]
clean-venv:
    rm -f uv.lock
    rm -rf .venv


[group("workflow")]
lint:
    uv run codespell src tests docs/source
    uv run ruff check --fix src tests
    uv run ruff format src tests
    uv run basedpyright src tests


[group("clean")]
clean-lint:
    rm -rf .ruff_cache


[group("workflow")]
test *FLAGS:
    uv sync --group test
    uv run pytest {{FLAGS}}


[group("workflow")]
test-fast *FLAGS:
    uv sync --group test
    uv run pytest -m "not slow_sim" {{FLAGS}}


[group("workflow")]
coverage:
    uv run coverage run -m pytest tests
    uv run coverage html
    uv run coverage report


[group("clean")]
clean-test:
    rm -rf .hypothesis
    rm -rf .pytest_cache
    rm -f .coverage
    rm -rf htmlcov


[group("workflow")]
build *FLAGS:
    uv build --refresh {{FLAGS}}


[group("clean")]
clean-build:
    rm -rf dist
    rm -rf src/circuit_benchmarks_guppy.egg-info


[group("workflow")]
commit *FLAGS:
    uv run cz commit {{FLAGS}}


[group("workflow")]
release:
    uv run semantic-release -v --noop version


[group("workflow")]
release-real:
    uv run semantic-release -v version



################################################################################
## Docs
################################################################################

docs-image := "docs-generator"
docs-container := "docs-generator-container"
docs-output := "docs/generated/_html"


[group("docs")]
docs-build-local-image:
    docker build -f docs/Dockerfile --target mount-local -t {{docs-image}} .


[group("docs")]
docs-build-ci-image:
    # Omit "--target mount-local" in order to build the docker image through the
    # last step, which copies the project into the docker image
    docker build -f docs/Dockerfile -t {{docs-image}} .


[group("docs")]
docs-clean-container:
    docker rm -f {{docs-container}} 2>/dev/null || true


[group("docs")]
docs-create-local-container:
    # Bind the local root directory
    docker create \
        --name {{docs-container}} \
        --mount type=bind,source="$(pwd)/src",target=/app/src \
        --mount type=bind,source="$(pwd)/notebooks",target=/app/notebooks \
        --mount type=bind,source="$(pwd)/README.md",target=/app/README.md \
        --mount type=bind,source="$(pwd)/docs",target=/app/docs \
        -w /app \
        {{docs-image}}


[group("docs")]
docs-create-ci-container:
    docker create \
        --name {{docs-container}} \
        -w /app \
        {{docs-image}}


[group("docs")]
docs-run-container:
    docker start {{docs-container}}
    docker exec {{docs-container}} bash -c \
        " \
            set -e && \
            rm -rf docs/generated/autoapi && \
            rm -rf docs/generated/_build/html && \
            python docs/src/_scripts/link_notebooks_folder.py notebooks/ && \
            sphinx-apidoc -o docs/generated/autoapi src/circuit_benchmarks_guppy --remove-old --separate --implicit-namespaces --no-toc -d 2 && \
            python docs/src/_scripts/format_index_files.py docs/generated/autoapi/ && \
            sphinx-build docs {{docs-output}} -v -b html \
        "

[group("workflow")]
[group("docs")]
docs-local:
    just docs-clean-container
    just docs-build-local-image
    just docs-create-local-container
    just docs-run-container
    docker rm -f {{docs-container}}


[group("docs")]
docs-ci:
    just docs-clean-container
    just docs-build-ci-image
    just docs-create-ci-container
    just docs-run-container

    # Remove old output if exists
    rm -rf {{docs-output}}

    # Copy the generated docs out of the container
    docker cp {{docs-container}}:/app/{{docs-output}} ./{{docs-output}}

    docker rm -f {{docs-container}}


[group("docs")]
docs-show:
    open {{docs-output}}/index.html


[group("clean")]
[group("docs")]
clean-docs:
    # Remove any old container with the same name (ignores error if not exists)
    docker rm -f {{docs-container}} 2>/dev/null || true

    # Remove artifacts from local directories
    rm -f docs/Makefile
    rm -f docs/make.bat
    rm -rf docs/generated/autoapi
    rm -rf docs/generated/notebooks
    rm -rf docs/generated/_build/html
    rm -rf docs/generated/_html



################################################################################
## Clean
################################################################################

[group("clean")]
clean:
    just clean-venv clean-lint clean-test clean-build clean-docs
