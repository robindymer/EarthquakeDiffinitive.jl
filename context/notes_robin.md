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

## Reuse K between runs
export EQD_STIFFNESS_CACHE=$HOME/.cache/eqd-stiffness

Set it once and every `build_model` reads/writes K there — a hit skips the
elastic assembly and all the CG solves. Unset = no caching (that is what CI
does). Build one offline, or list what is cached:

julia --project=scripts -t auto scripts/build_stiffness_cache.jl 20 1600 1200 exact
julia --project=scripts scripts/build_stiffness_cache.jl --list

## TODOs / thoughts
- Order of FD used?
- Theory of the iterative method?
- CG indirect method, used to solve large Ax=b systems
- Cholesky is a direct method, probably not viable for full scale problem

## Running on UPPMAX
module load Julia/1.11.3-linux-x86_64
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using EarthquakeDiffinitive; println("ok")'

./scripts/submit_bp8.sh 20
./scripts/submit_bp8.sh 10
