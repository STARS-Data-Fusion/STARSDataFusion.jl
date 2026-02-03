@testset "Spatial Utilities" begin
    @testset "find_nearest_ij" begin
        origin = [0.0, 0.0]
        cell_size = [1.0, 1.0]
        
        # Test basic cases
        @test STARSDataFusion.find_nearest_ij([0.0, 0.0], origin, cell_size) == [1, 1]
        @test STARSDataFusion.find_nearest_ij([1.0, 1.0], origin, cell_size) == [2, 2]
        @test STARSDataFusion.find_nearest_ij([0.5, 0.5], origin, cell_size) == [1, 1]
        @test STARSDataFusion.find_nearest_ij([1.5, 2.5], origin, cell_size) == [2, 3]
        
        # Test with negative cell size
        cell_size_neg = [1.0, -1.0]
        result_neg = STARSDataFusion.find_nearest_ij([1.0, -1.0], origin, cell_size_neg)
        @test isa(result_neg, Vector{Int64})
        @test length(result_neg) == 2
        
        # Test with offset origin
        origin_offset = [10.0, 20.0]
        @test STARSDataFusion.find_nearest_ij([10.0, 20.0], origin_offset, cell_size) == [1, 1]
        @test STARSDataFusion.find_nearest_ij([11.0, 21.0], origin_offset, cell_size) == [2, 2]
    end
    
    @testset "find_nearest_ij_multi" begin
        origin = [0.0, 0.0]
        cell_size = [1.0, 1.0]
        n_dims = [10, 10]
        
        # Test with multiple targets
        targets = [0.5 0.5; 1.5 1.5; 2.5 2.5]
        result = STARSDataFusion.find_nearest_ij_multi(targets, origin, cell_size, n_dims)
        
        @test size(result, 2) == 2  # Should have 2 columns (i, j)
        @test all(result[:, 1] .<= n_dims[1])
        @test all(result[:, 2] .<= n_dims[2])
        @test all(result[:, 1] .>= 1)
        @test all(result[:, 2] .>= 1)
        
        # Test filtering of out-of-bounds points
        targets_oob = [0.5 0.5; 15.0 15.0; 2.5 2.5]
        result_oob = STARSDataFusion.find_nearest_ij_multi(targets_oob, origin, cell_size, n_dims)
        # Should have fewer points than input (out-of-bounds filtered)
        @test size(result_oob, 1) < size(targets_oob, 1)
    end
    
    @testset "find_nearest_ind" begin
        # Test basic index finding
        @test STARSDataFusion.find_nearest_ind(0.0, 0.0, 1.0) == 1
        @test STARSDataFusion.find_nearest_ind(1.0, 0.0, 1.0) == 2
        @test STARSDataFusion.find_nearest_ind(0.5, 0.0, 1.0) == 2
        @test STARSDataFusion.find_nearest_ind(2.5, 0.0, 1.0) == 4  # round(2.5) = 3, add 1 = 4
        
        # Test with different cell size
        @test STARSDataFusion.find_nearest_ind(5.0, 0.0, 2.0) == 4  # round(2.5) = 3, add 1 = 4
        
        # Test with offset origin
        @test STARSDataFusion.find_nearest_ind(10.0, 5.0, 1.0) == 6
    end
    
    @testset "cell_size" begin
        # Test with 2D raster
        raster_2d = Raster(rand(10, 10), dims=(X(0.0:1.0:9.0), Y(0.0:2.0:18.0)))
        width, height = cell_size(raster_2d)
        
        @test width isa Number
        @test height isa Number
        @test width > 0
        @test abs(height) > 0  # Height can be negative depending on direction
        
        # Test with 3D raster
        raster_3d = Raster(rand(10, 10, 3), dims=(X(0.0:1.0:9.0), Y(0.0:2.0:18.0), Ti(1:3)))
        width_3d, height_3d = cell_size(raster_3d)
        
        @test width_3d isa Number
        @test height_3d isa Number
    end
    
    @testset "get_centroid_origin_raster" begin
        # Create a simple raster
        raster = Raster(rand(5, 5), dims=(X(0.0:1.0:4.0), Y(0.0:1.0:4.0)))
        
        centroid, origin = get_centroid_origin_raster(raster)
        
        # Check that centroid and origin are numeric (based on actual function behavior)
        @test centroid isa Number
        @test origin isa Number
        
        # Check reasonable values
        @test centroid >= origin
    end
    
    @testset "get_x_matrix and get_y_matrix" begin
        # Create a simple raster
        raster = Raster(rand(3, 4), dims=(X([1.0, 2.0, 3.0]), Y([10.0, 20.0, 30.0, 40.0])))
        
        # Test get_x_matrix
        x_matrix = STARSDataFusion.get_x_matrix(raster)
        @test size(x_matrix) == (3, 4)
        # Each column should be the same
        @test all(x_matrix[:, 1] .== x_matrix[:, 2])
        # Values should match X dimension
        @test x_matrix[:, 1] == [1.0, 2.0, 3.0]
        
        # Test get_y_matrix
        y_matrix = STARSDataFusion.get_y_matrix(raster)
        @test size(y_matrix) == (3, 4)
        # Each row should be the same
        @test all(y_matrix[1, :] .== y_matrix[2, :])
        # Values should match Y dimension
        @test y_matrix[1, :] == [10.0, 20.0, 30.0, 40.0]
    end
    
    @testset "get_missing_indices" begin
        # Test with NaN missing value
        raster_nan = Raster([1.0 2.0; NaN 4.0; 5.0 NaN], dims=(X(1:3), Y(1:2)), missingval=NaN)
        missing_inds = STARSDataFusion.get_missing_indices(raster_nan)
        @test sum(missing_inds) == 2  # Two NaN values
        
        # Test with specific missing value
        raster_missing = Raster([1.0 2.0; -9999.0 4.0; 5.0 -9999.0], dims=(X(1:3), Y(1:2)), missingval=-9999.0)
        missing_inds2 = STARSDataFusion.get_missing_indices(raster_missing)
        @test sum(missing_inds2) == 2  # Two missing values
    end
end
