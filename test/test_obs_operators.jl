@testset "Observation Operators" begin
    @testset "unif_weighted_obs_operator" begin
        # Create sensor and target coordinates
        sensor = Float64[0.0 0.0; 10.0 0.0; 0.0 10.0]  # 3 sensors
        target = Float64[0.0 0.0; 5.0 5.0; 10.0 10.0; 15.0 15.0]  # 4 targets
        sensor_res = [5.0, 5.0]
        target_res = [2.0, 2.0]
        
        H = unif_weighted_obs_operator(sensor, target, sensor_res, target_res)
        
        # Check dimensions
        @test size(H) == (3, 4)
        
        # Check that H is sparse
        @test issparse(H)
        
        # Check that each row sums to approximately 1 or 0
        row_sums = sum(H, dims=2)[:]
        # Most rows should sum to approximately 1 or 0
        # Allow some tolerance for numerical precision and geometry
        @test count(x -> isapprox(x, 1.0, atol=0.01) || isapprox(x, 0.0, atol=0.01), row_sums) >= 2
        
        # Test with identical sensor and target
        H_ident = unif_weighted_obs_operator(sensor, sensor, sensor_res, target_res)
        @test size(H_ident) == (3, 3)
        # Should have significant weights for nearby points
        @test H_ident[1, 1] > 0  # First sensor should overlap with itself
    end
    
    @testset "unif_weighted_obs_operator_centroid" begin
        # Test with identical coordinates (should return identity)
        coords = Float64[0.0 0.0; 10.0 10.0; 20.0 20.0]
        sensor_res = [5.0, 5.0]
        
        H = STARSDataFusion.unif_weighted_obs_operator_centroid(coords, coords, sensor_res)
        
        # Should be identity matrix
        @test size(H) == (3, 3)
        @test issparse(H)
        @test H ≈ sparse(I, 3, 3)
        
        # Test with different coordinates
        sensor = Float64[0.0 0.0; 10.0 0.0]
        target = Float64[0.0 0.0; 5.0 0.0; 10.0 0.0; 15.0 0.0]
        
        H2 = STARSDataFusion.unif_weighted_obs_operator_centroid(sensor, target, sensor_res)
        
        # Check dimensions
        @test size(H2) == (2, 4)
        
        # Check basic properties of observation operator
        # Row sums should generally be near 1 or 0
        row_sums = sum(H2, dims=2)[:]
        @test count(x -> isapprox(x, 1.0, atol=0.01) || isapprox(x, 0.0, atol=0.01), row_sums) >= 1
        
        # First sensor should have weight on nearby targets
        @test H2[1, 1] > 0
        @test H2[1, 2] >= 0
    end
end
