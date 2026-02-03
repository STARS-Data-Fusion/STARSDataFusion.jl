@testset "Utility Functions" begin
    @testset "nanmean" begin
        # Test with no NaNs
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test nanmean(x) ≈ 3.0
        
        # Test with NaNs
        x_nan = [1.0, 2.0, NaN, 4.0, 5.0]
        @test nanmean(x_nan) ≈ 3.0
        
        # Test with all NaNs
        all_nan = [NaN, NaN, NaN]
        @test isnan(nanmean(all_nan))
        
        # Test with matrix (dimension argument)
        mat = [1.0 2.0 3.0; 4.0 5.0 6.0]
        result = nanmean(mat, 1)
        @test size(result) == (1, 3)
        @test result[1] ≈ 2.5
        @test result[2] ≈ 3.5
        @test result[3] ≈ 4.5
    end
    
    @testset "nanvar" begin
        # Test with no NaNs
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        @test nanvar(x) ≈ var([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with NaNs
        x_nan = [1.0, 2.0, NaN, 4.0, 5.0]
        @test nanvar(x_nan) ≈ var([1.0, 2.0, 4.0, 5.0])
        
        # Test with all NaNs
        all_nan = [NaN, NaN, NaN]
        @test isnan(nanvar(all_nan))
    end
    
    @testset "col_major and row_major" begin
        # Test column major indexing
        nrow, ncol = 3, 4
        @test STARSDataFusion.col_major(1, 1, nrow) == 1
        @test STARSDataFusion.col_major(3, 1, nrow) == 3
        @test STARSDataFusion.col_major(1, 2, nrow) == 4
        @test STARSDataFusion.col_major(2, 2, nrow) == 5
        
        # Test row major indexing
        @test STARSDataFusion.row_major(1, 1, ncol) == 1
        @test STARSDataFusion.row_major(1, 2, ncol) == 2
        @test STARSDataFusion.row_major(2, 1, ncol) == 5
    end
    
    @testset "filter_val and replace_val" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test filter_val
        filtered = STARSDataFusion.filter_val(x, 2.0, 4.0)
        @test filtered == [2.0, 3.0, 4.0]
        
        # Test replace_val
        replaced = STARSDataFusion.replace_val(copy(x), 2.0, 4.0, -999.0)
        @test replaced[1] == -999.0
        @test replaced[2] == 2.0
        @test replaced[5] == -999.0
    end
end