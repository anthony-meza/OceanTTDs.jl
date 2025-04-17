using TCMGreensFunctions
using Documenter

DocMeta.setdocmeta!(TCMGreensFunctions, :DocTestSetup, :(using TCMGreensFunctions); recursive=true)

makedocs(;
    modules=[TCMGreensFunctions],
    authors="A Meza",
    sitename="TCMGreensFunctions.jl",
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
