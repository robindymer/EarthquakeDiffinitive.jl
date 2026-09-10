# Free/total VRAM, across CUDA.jl versions.
#
# `CUDA.available_memory()` exists in CUDA.jl 5.x but not in 6.x, where the
# package split into CUDACore/CUDATools and only `memory_info()` survived —
# and `scripts/Project.toml` allows `CUDA = "5, 6"`. Both call sites here run
# *after* an hour of host assembly, so an `UndefVarError` on a progress print
# is an expensive way to learn which version got resolved.
#
# `memory_info()` returns `(free, total)` in both, so prefer it and keep the
# 5.x names only as a fallback.
if isdefined(CUDA, :memory_info)
    vram_free()  = Int(CUDA.memory_info()[1])
    vram_total() = Int(CUDA.memory_info()[2])
else
    vram_free()  = Int(CUDA.available_memory())
    vram_total() = Int(CUDA.total_memory())
end
