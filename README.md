# circuit-benchmarks-guppy
A repository for benchmarking of quantum gates and circuits, written in Guppy

## Getting Started

- Install guppy (this will also install selene_sim, which is needed for simulations):

```
pip install guppylang
```

- Install qnexus (for submission to Quantinuum hardware or emulator backends):

```
pip install qnexus
```

- Optional: for access to hardware realistic noise model that uses the compiler:

-Install selene-anduril
Located in the Quantinuum Artifactory here: https://quantinuumsw.jfrog.io/ui/packages/pypi:%2F%2Fselene-anduril?name=selene-anduril&type=packages

-Install pecos-selene
Located in the Quantinuum Artifactory here: https://quantinuumsw.jfrog.io/ui/packages/pypi:%2F%2Fpecos-selene?name=pecos-selene&type=packages
