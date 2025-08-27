# circuit-benchmarks-guppy

A repository for benchmarking of quantum gates and circuits, written in Guppy

## Installation

By default `circuit-benchmarks-guppy` only creates circuits. In order to run circuits as a simulation, on an emulator, or submit them to hardware, you will need to install an additional set of dependencies. The following execution modes and their dependencies are supported:

- Simulation
    - Required, dependency group `sim`
        - `selene-sim` through PyPi
    - Optional, dependency group `sim-noise`, all through [quantinuumsw JFrog Artifactory](https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple)
        - `selene-anduril`
        - `pecos-selene` 
        - `selene_custom_error_model`

- Emulation
    - Required, dependency group `hardware`
        - `qnexus` through PyPi

- Hardware
    - Required, dependency group `hardware`
        - `qnexus` through PyPi


`circuit-benchmarks-guppy` is not currently packaged, and needs to be installed via its git repo. Installing all execution dependencies looks like the following. Delete the groups that are not needed.

From Github:

```sh
pip install "circuit-benchmarks-guppy[sim,sim-noise,hardware] @ git+ssh://git@github.com/CQCL/circuit-benchmarks-guppy@main"
```

Or Bitbucket:

```sh
pip install "circuit-benchmarks-guppy[sim,sim-noise,hardware] @ git+ssh://git@co41-bitbucket.honeywell.lab:7999/theor/circuit-benchmarks-guppy@main"
```

Replace `pip install` with `uv add` to add it as a dependency.


Installing the optional simulation noise dependencies requires access to the [`quantinuumsw` JFrog Artifactory](https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple). Submit an IT ticket to request access.


## Contributing

Access to the [quantinuumsw JFrog Artifactory](https://quantinuumsw.jfrog.io/artifactory/api/pypi/pypi_local/simple) repository is mandatory for contributions that change the dependencies.

This package uses the `uv` package manager and the `just` orchestration tool.

Both of these tools are most easily installed using `pipx`: [Installation guide](https://pipx.pypa.io/stable/installation/).


- `pipx install uv`, alternatively: [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- `pipx install rust-just`, alternativly: [Installation guide](https://just.systems/man/en/packages.html).

