# Helmholtz-based Adjoint method

Simulation code used in "Fast free-form phase mask design for three-dimensional photolithography using convergent Born series".

- target material: PDMS + TiO2 + Microchem SU-8 2000 layered material
- test code: Vanilla inverse solver/ADJOINT_EXAMPLE.m

## prerequisite

- matlab (Test in 2022b - 2024b)
- Parallel computering Toolbox
- Optimization Toolbox
- Signal Processing Toolbox
- Curve Fitting Toolbox
- [yaml](https://kr.mathworks.com/matlabcentral/fileexchange/106765-yaml?s_tid=FX_rc3_behav)

## installation

[git](https://git-scm.com/downloads) is required to download and call database. Then, type the following commands in powershell(windows) or terminal(linux): 

```bash
cd "~~~" # type the absolute path of directory to store the repository
# loading main repository
git clone https://github.com/ehgus/Helmholtz-adjoint-solver.git
cd Helmholtz-adjoint-solver
# loading refractive index database
git submodule init 
git submodule update
```
You should have matlab and required packages to execute test suites.

## Name convention

The name convention refers to julia's reference style guide.

- module, class: capitalization and camel case: `ForwardSimulation`. When a word is next to the abbreivation, you can use small word (`FDTDsolver`).
- function, method: lowercase with multiple words squashed together (`isequal`,`haskey`). When necessary, use underscores as word separators.
- conciseness is valued, but avoid abbreviation (`index` rather than `i`).
- It is allowed to use capital letter exceptionally when using a proper noun or a abbreivation.
 