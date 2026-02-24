using LaplaceMFS
using Documenter

DocMeta.setdocmeta!(LaplaceMFS, :DocTestSetup, :(using LaplaceMFS); recursive=true)

makedocs(;
    modules=[LaplaceMFS],
    authors="Xuanzhao Gao <xgao@flatironinstitute.org> and contributors",
    sitename="LaplaceMFS.jl",
    format=Documenter.HTML(;
        canonical="https://ArrogantGao.github.io/LaplaceMFS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ArrogantGao/LaplaceMFS.jl",
    devbranch="main",
)
