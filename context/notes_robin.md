## Get main repo working
registry up
activate .
update

## Make docs
activate docs
update
instantiate

## Start pluto
- (if not yet installed) import Pkg; Pkg.add("Pluto")
- import Pluto; Pluto.run()

## Run script
julia --project=scripts scripts/elastic_wave_2d.jl

## TODOs / thoughts
- Order of FD used?
- Theory of the iterative method?
- CG indirect method, used to solve large Ax=b systems
- Cholesky is a direct method, probably not viable for full scale problem
