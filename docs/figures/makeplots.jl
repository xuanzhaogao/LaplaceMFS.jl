using CairoMakie, DelimitedFiles, Printf

const SP = joinpath(@__DIR__, "data")

# ---- read a headered csv into (header, Dict{String,Vector{Float64}}) ----
function readcsv(path)
    raw, hdr = readdlm(path, ',', Float64, '\n'; header = true)
    cols = Dict{String,Vector{Float64}}()
    for (j, name) in enumerate(vec(hdr))
        cols[strip(String(name))] = raw[:, j]
    end
    return cols
end

# ---- validated categorical palette (dataviz reference instance) ----
const LIGHT = (
    surface = "#fcfcfb", panel = "#fcfcfb",
    ink = "#0b0b0b", ink2 = "#52514e", ink3 = "#8a8880",
    grid = "#e6e5e1", axis = "#c8c6c0",
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    ref = "#8a8880",
)
const DARK = (
    surface = "#1a1a19", panel = "#1a1a19",
    ink = "#ffffff", ink2 = "#c3c2b7", ink3 = "#8f8e86",
    grid = "#2f2f2c", axis = "#494844",
    series = ["#3987e5", "#d95926", "#199e70", "#c98500"],
    ref = "#8f8e86",
)

const SUPS = Dict('0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴',
                  '5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹','-'=>'⁻')
sup(n::Int) = join(SUPS[c] for c in string(n))
decades(ps) = ([10.0^p for p in ps], ["10" * sup(p) for p in ps])

function build(T, outfile)
    fig = Figure(size = (1020, 800), backgroundcolor = T.surface,
                 figure_padding = (18, 26, 14, 18))

    axkw = (
        backgroundcolor = T.panel,
        xgridcolor = T.grid, ygridcolor = T.grid,
        xgridwidth = 1, ygridwidth = 1,
        leftspinecolor = T.axis, bottomspinecolor = T.axis,
        topspinevisible = false, rightspinevisible = false,
        xtickcolor = T.axis, ytickcolor = T.axis,
        xticklabelcolor = T.ink2, yticklabelcolor = T.ink2,
        xlabelcolor = T.ink2, ylabelcolor = T.ink2,
        titlecolor = T.ink, subtitlecolor = T.ink2,
        xticklabelsize = 12, yticklabelsize = 12,
        xlabelsize = 13, ylabelsize = 13,
        titlesize = 15, subtitlesize = 12,
        titlealign = :left, titlefont = :bold,
    )

    # ============ panel 1: convergence vs proxy degree ============
    d1 = readcsv("$SP/f1_convergence.csv")
    ax1 = Axis(fig[1, 1]; yscale = log10,
        title = "Accuracy is set by proxy degree",
        subtitle = "single sphere vs exact Legendre series, r_p = 0.7",
        xlabel = "proxy degree  t ≈ √(2N) − 1", ylabel = "relative error", axkw...)
    for (i, (da, dy)) in enumerate(((3.0, -5), (2.0, 5), (1.5, 0), (1.2, 0)))
        m = d1["da"] .== da
        t, e = d1["t"][m], d1["relerr"][m]
        lines!(ax1, t, e; color = T.series[i], linewidth = 2)
        scatter!(ax1, t, e; color = T.series[i], markersize = 9,
                 strokecolor = T.panel, strokewidth = 1.5)
        text!(ax1, t[end], e[end]; text = "d/a = $(da)", color = T.ink2,
              fontsize = 11.5, align = (:left, :center), offset = (8, dy))
    end
    xlims!(ax1, 17.5, 58); ylims!(ax1, 2e-10, 4e-1)
    ax1.yticks = decades(-9:2:-1)

    # ============ panel 2: residual is not the error ============
    d2 = readcsv("$SP/f2_rp.csv")
    ax2 = Axis(fig[1, 2]; yscale = log10,
        title = "The residual is not the error",
        subtitle = "d/a = 1.2, M = 723, N = 614",
        xlabel = "proxy radius  r_p / a", ylabel = "relative magnitude", axkw...)
    lines!(ax2, d2["r_p"], d2["relerr"]; color = T.series[1], linewidth = 2)
    scatter!(ax2, d2["r_p"], d2["relerr"]; color = T.series[1], markersize = 9,
             strokecolor = T.panel, strokewidth = 1.5)
    lines!(ax2, d2["r_p"], d2["resid"]; color = T.series[2], linewidth = 2)
    scatter!(ax2, d2["r_p"], d2["resid"]; color = T.series[2], markersize = 9,
             marker = :rect, strokecolor = T.panel, strokewidth = 1.5)
    ib = argmin(d2["resid"]); ie = argmin(d2["relerr"])
    ratio = d2["relerr"][ib] / d2["relerr"][ie]
    scatter!(ax2, [d2["r_p"][ie]], [d2["relerr"][ie]]; color = :transparent,
             markersize = 17, strokecolor = T.series[1], strokewidth = 1.5)
    text!(ax2, d2["r_p"][ie], d2["relerr"][ie]; text = "best error",
          color = T.ink2, fontsize = 10.5, align = (:center, :top),
          offset = (0, -14))
    text!(ax2, 0.615, 0.16;
          text = @sprintf("the residual falls monotonically\nwhile the error blows up —\n%.0f× worse at r_p = %.2f",
                          ratio, d2["r_p"][ib]),
          color = T.ink2, fontsize = 10.5, align = (:center, :center))
    ax2.yticks = decades(-3:1:0)
    axislegend(ax2, [
            [LineElement(color = T.series[1], linewidth = 2),
             MarkerElement(color = T.series[1], marker = :circle, markersize = 9)],
            [LineElement(color = T.series[2], linewidth = 2),
             MarkerElement(color = T.series[2], marker = :rect, markersize = 9)],
        ], ["field error", "boundary residual"];
        position = :lt, framevisible = false, labelcolor = T.ink2,
        labelsize = 11.5, patchsize = (22, 12), rowgap = 1)

    # ============ panel 3: many-body far-field decay ============
    d3 = readcsv("$SP/f3_decay.csv")
    ax3 = Axis(fig[2, 1]; yscale = log10, xscale = log10,
        title = "Far-field decay follows O(L⁻⁴)",
        subtitle = "vs exact single-sphere answer as neighbours recede",
        xlabel = "separation  L / a", ylabel = "relative error", axkw...)
    m2, m27 = d3["nspheres"] .== 2, d3["nspheres"] .== 27
    # short slope gauge, parked in the empty upper-right where both curves have gone
    Lg = [30.0, 64.0]; g0 = 1.2e-2
    lines!(ax3, Lg, g0 .* (Lg ./ Lg[1]) .^ (-4); color = T.ref, linewidth = 1.5,
           linestyle = :dash)
    text!(ax3, 44.0, g0 * (44.0/30.0)^(-4); text = "slope −4", color = T.ink3,
          fontsize = 11, align = (:left, :bottom), offset = (7, 5))
    for (i, (m, lbl)) in enumerate(((m2, "2 spheres"), (m27, "27 spheres")))
        L, e = d3["L"][m], d3["relerr"][m]
        lines!(ax3, L, e; color = T.series[i], linewidth = 2)
        scatter!(ax3, L, e; color = T.series[i], markersize = 9,
                 marker = i == 1 ? :circle : :rect,
                 strokecolor = T.panel, strokewidth = 1.5)
        text!(ax3, L[end], e[end]; text = "  " * lbl, color = T.ink2,
              fontsize = 11.5, align = (:left, :center))
    end
    xlims!(ax3, 5.5, 175)
    ax3.xticks = ([6, 12, 24, 48, 96], ["6", "12", "24", "48", "96"])
    ax3.yticks = decades(-7:1:-2)

    # ============ panel 4: iteration count vs sphere count ============
    d4 = readcsv("$SP/f4_scaling.csv")
    ax4 = Axis(fig[2, 2]; xscale = log10,
        title = "GMRES iterations stay flat",
        subtitle = "cubic lattice, 1a surface gaps, rtol = 1e-10",
        xlabel = "spheres", ylabel = "GMRES iterations", axkw...)
    lines!(ax4, d4["nspheres"], d4["iters"]; color = T.series[1], linewidth = 2)
    scatter!(ax4, d4["nspheres"], d4["iters"]; color = T.series[1], markersize = 10,
             strokecolor = T.panel, strokewidth = 1.5)
    for k in eachindex(d4["nspheres"])
        u = d4["unknowns"][k]
        lab = u >= 1000 ? @sprintf("%.0fk", u/1000) : @sprintf("%.0f", u)
        text!(ax4, d4["nspheres"][k], d4["iters"][k]; text = lab, color = T.ink3,
              fontsize = 10, align = (:center, :bottom), offset = (0, 9))
    end
    ylims!(ax4, 0, 13)
    ax4.xticks = ([8, 27, 64, 125, 216, 512],
                  ["8", "27", "64", "125", "216", "512"])
    text!(ax4, 8.6, 1.1; text = "labels: number of unknowns", color = T.ink3,
          fontsize = 10.5, align = (:left, :bottom))

    Label(fig[0, 1:2], "LaplaceMFS verification — single sphere to 512 spheres";
          color = T.ink, fontsize = 18, font = :bold, halign = :left,
          padding = (2, 0, 0, 4))
    rowgap!(fig.layout, 1, 18)
    colgap!(fig.layout, 1, 30)
    rowgap!(fig.layout, 2, 26)

    save(outfile, fig; px_per_unit = 2)
    println("wrote $outfile")
    return fig
end

build(LIGHT, "$(@__DIR__)/mfs_verification_light.png")
build(DARK,  "$(@__DIR__)/mfs_verification_dark.png")
build(LIGHT, "$(@__DIR__)/mfs_verification.pdf")
