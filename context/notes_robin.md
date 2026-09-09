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
julia -e 'using Pkg
          Pkg.Registry.add(RegistrySpec(
              url="https://github.com/Diffinitive/diffinitive_registry"))'

module load Julia/1.11.3-linux-x86_64
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using EarthquakeDiffinitive; println("ok")'

./scripts/submit_bp8.sh 20
./scripts/submit_bp8.sh 10 1200 1200 880

## Misc
[robind@pelle1 scripts]$ ./submit_bp8.sh 20
BP8-QD-GS submission chain
  repo         /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl
  dz           20 m
  domain       L_fault = 1600 m, L_normal = 1200 m
  shards       32  (8 cores, 24G each)
  K cache      /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/eqd-stiffness
  account      uppmax2026-1-45   partition pelle
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [1] K shards      job 6678514  (array 1-32)
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [2] merge         job 6678515  (after 6678514)
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [3] run + outputs job 6678516  (after 6678515)

Submitted. Watch with:  squeue -u $USER
Outputs will appear in: /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/output/BP8-QD-GS_dz20_Lf1600_Ln1200_exact/
Logs in:                /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/logs/
[robind@pelle1 scripts]$ ./submit_bp8.sh 10 1200 1200 880
BP8-QD-GS submission chain
  repo         /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl
  dz           10 m
  domain       L_fault = 1200 m, L_normal = 1200 m
  shards       880  (16 cores, 96G each)
  K cache      /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/eqd-stiffness
  account      uppmax2026-1-45   partition pelle
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [1] K shards      job 6678517  (array 1-880)
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [2] merge         job 6678518  (after 6678517)
sbatch: --mail-user is not supported on Pelle
sbatch: Mail will be sent to your e-mail address in SUPR
  [3] run + outputs job 6678519  (after 6678518)

Submitted. Watch with:  squeue -u $USER
Outputs will appear in: /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/output/BP8-QD-GS_dz10_Lf1200_Ln1200_exact/
Logs in:                /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/logs/
[robind@pelle1 scripts]$