@testset "Point Tests" begin
    using STARSDataFusion.BBoxes.Points
    
    @testset "Point Constructor" begin
        # Test basic constructor
        p = Point(10.0, 20.0)
        @test p.x == 10.0
        @test p.y == 20.0
        
        # Test with negative coordinates
        p_neg = Point(-5.0, -10.0)
        @test p_neg.x == -5.0
        @test p_neg.y == -10.0
        
        # Test with zero coordinates
        p_zero = Point(0.0, 0.0)
        @test p_zero.x == 0.0
        @test p_zero.y == 0.0
    end
    
    # Note: Tests for ArchGDAL point conversion functions (from_AG, to_AG, etc.)
    # would require ArchGDAL setup and are better suited for integration tests
end
