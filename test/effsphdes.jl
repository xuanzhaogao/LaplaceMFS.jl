using LaplaceMFS
using Test

@testset "EffSphDes" begin
    points = load_sphdes_t(10)
    @test size(points) == (62, 3)
    @test all(isapprox.(sqrt.(sum(abs2, points; dims=2)), 1.0; atol=1e-12))
    @test_throws ArgumentError load_sphdes_t(0)
    @test_throws ArgumentError load_sphdes_t(181)

    d = sphdes_num_points()
    @test length(d) == 180
    @test d[1] == 3
    @test d[10] == 62
    @test d[180] == 16382
    @test size(load_sphdes_N(d[170]), 1) == d[170]
end
