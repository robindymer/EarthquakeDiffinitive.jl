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
