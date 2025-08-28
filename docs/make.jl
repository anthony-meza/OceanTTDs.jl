using OceanTTDs
using Documenter

DocMeta.setdocmeta!(OceanTTDs, :DocTestSetup, :(using OceanTTDs); recursive=true)

makedocs(;
    modules=[OceanTTDs],
    authors="A Meza",
    sitename="OceanTTDs.jl",
    format=Documenter.HTML(;
        canonical="https://anthony-meza.github.io/TCMGreensFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anthony-meza/TCMGreensFunctions.jl",
    devbranch="main",
)
