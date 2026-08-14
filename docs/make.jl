using EarthquakeDiffinitive
using Documenter

DocMeta.setdocmeta!(EarthquakeDiffinitive, :DocTestSetup, :(using EarthquakeDiffinitive); recursive=true)

makedocs(;
    modules=[EarthquakeDiffinitive],
    authors="Robin Dymér <robin.dymer@hotmail.com> and contributors",
    sitename="EarthquakeDiffinitive.jl",
    format=Documenter.HTML(;
        canonical="https://robindymer.github.io/EarthquakeDiffinitive.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/robindymer/EarthquakeDiffinitive.jl",
    devbranch="main",
)
