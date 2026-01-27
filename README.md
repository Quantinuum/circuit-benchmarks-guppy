# Solarium

A repository for benchmarking of quantum machines, written in Guppy.

## Installation

The `solarium` package creates quantum circuits, runs those circuits on a backend, and analyses the resulting data. In order to run, a backend must be specified. There are two backends:

1. Selene, which allows simulation of circuit results.

    Install by including the optional `sim` dependency group:

    ```bash
    pip install solarium[sim]
    ```

    More complicated noise model can be installed as well:

    ```bash
    pip install solarium[sim,sim-noise]
    ```

2. QNexus, which allows the submission of jobs to run on hardware emulators or the hardware itself.

    Install by including the optional `sim` dependency group:

    ```bash
    pip install solarium[hardware]
    ```


Note! Solarium currently relies on internal-exclusive tools and installation will break if you do not have access to SW Artifactory.

Simulation without advanced noise models is accessible via the publicly available `selene` package. Running circuits on emulators or on hardware requires a Nexus account and plan.

Solarium is not currently distributed, and needs to be installed via its git repo. Installing all execution dependencies looks like the following. Delete the groups that are not needed.

```sh
pip install "solarium[sim,sim-noise,hardware] @ git+ssh://git@github.com/Quantinuum/circuit-benchmarks-guppy@main"
```

Installing the optional simulation noise dependencies requires access to the [`quantinuumsw` JFrog Artifactory](https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple). Submit an IT ticket to request access.


## Contributing

Access to the [quantinuumsw JFrog Artifactory](https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple) repository is mandatory for contributions that change the dependencies.

This package uses the `uv` package manager and the `just` orchestration tool.

Both of these tools are most easily installed using `pipx`: [Installation guide](https://pipx.pypa.io/stable/installation/).

- `pipx install uv`, alternatively: [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- `pipx install rust-just`, alternativly: [Installation guide](https://just.systems/man/en/packages.html).

