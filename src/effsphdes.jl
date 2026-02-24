# read the sphere design points from https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/

effsphdes_path() = joinpath(artifact"EffSphDes")

function load_sphdes_t(t::Integer)
    (t >= 1 && t <= 180) || throw(ArgumentError("t must be >= 1 and <= 180, got $t"))

    prefix = @sprintf("sf%03d.", t)
    matches = filter(name -> startswith(name, prefix), readdir(effsphdes_path()))
    isempty(matches) && throw(ArgumentError("no sphere points file found for t=$t"))
    length(matches) == 1 || throw(ArgumentError("multiple sphere points files found for t=$t"))

    points = readdlm(joinpath(effsphdes_path(), only(matches)), Float64)
    size(points, 2) == 3 || throw(ArgumentError("expected 3 columns (x,y,z), got $(size(points, 2))"))
    return Matrix{Float64}(points)
end

# build a dictionary,  key: t, value: N
function sphdes_num_points()
    d = Vector{Int}(undef, 180)
    for name in readdir(effsphdes_path())
        m = match(r"^sf(\d{3})\.(\d+)$", name)
        isnothing(m) && continue
        t = parse(Int, m.captures[1])
        n = parse(Int, m.captures[2])
        d[t] = n
    end
    return d
end

function load_sphdes_N(N::Integer)
    (N >= 3 && N <= 16382) || throw(ArgumentError("N must be >= 3 and <= 16382, got $N"))
    t = findfirst(x -> x == N, sphdes_num_points())
    isnothing(t) && throw(ArgumentError("no sphere points file found for N=$N"))
    return load_sphdes_t(t)
end