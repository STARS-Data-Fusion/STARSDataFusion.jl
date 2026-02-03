@testset "Data Structures" begin
    @testset "DataFusionState" begin
        # Create sample dimensions
        dims_fine = (X(1.0:0.1:2.0), Y(1.0:0.1:2.0))
        dims_coarse = (X(1.0:0.5:2.0), Y(1.0:0.5:2.0))
        
        # Create sample rasters
        mean_raster = Raster(rand(11, 11, 3), dims=(dims_fine..., Ti(1:3)))
        sd_raster = Raster(rand(11, 11, 3), dims=(dims_fine..., Ti(1:3)))
        bias_mean_raster = Raster(rand(3, 3, 3), dims=(dims_coarse..., Ti(1:3)))
        bias_sd_raster = Raster(rand(3, 3, 3), dims=(dims_coarse..., Ti(1:3)))
        
        # Test constructor without conditional simulations
        state = DataFusionState(mean_raster, sd_raster, bias_mean_raster, bias_sd_raster, nothing)
        @test size(state.mean) == (11, 11, 3)
        @test size(state.SD) == (11, 11, 3)
        @test size(state.mean_bias) == (3, 3, 3)
        @test size(state.SD_bias) == (3, 3, 3)
        @test isnothing(state.cond_sims)
        
        # Test constructor with conditional simulations
        cond_sims = Raster(rand(11, 11, 5, 3), dims=(dims_fine..., Z(1:5), Ti(1:3)))
        state_with_sims = DataFusionState(mean_raster, sd_raster, bias_mean_raster, bias_sd_raster, cond_sims)
        @test !isnothing(state_with_sims.cond_sims)
        @test size(state_with_sims.cond_sims) == (11, 11, 5, 3)
    end
    
    @testset "STARSInstrumentData" begin
        # Create test data
        data = rand(10, 5)  # 10 spatial locations, 5 time steps
        bias = 0.01
        uq = 0.001
        dynamic_bias = true
        dynamic_bias_coefs = [0.95, 0.0001]
        spatial_resolution = [30.0, 30.0]
        dates = collect(1:5)
        coords = rand(10, 2)
        
        # Test constructor
        inst_data = STARSInstrumentData(data, bias, uq, dynamic_bias, dynamic_bias_coefs, spatial_resolution, dates, coords)
        
        @test size(inst_data.data) == (10, 5)
        @test inst_data.bias == 0.01
        @test inst_data.uq == 0.001
        @test inst_data.dynamic_bias == true
        @test inst_data.dynamic_bias_coefs == [0.95, 0.0001]
        @test inst_data.spatial_resolution == [30.0, 30.0]
        @test length(inst_data.dates) == 5
        @test size(inst_data.coords) == (10, 2)
        
        # Test with array bias
        bias_array = rand(10, 5)
        inst_data_array = STARSInstrumentData(data, bias_array, uq, dynamic_bias, dynamic_bias_coefs, spatial_resolution, dates, coords)
        @test size(inst_data_array.bias) == (10, 5)
    end
    
    @testset "STARSInstrumentGeoData" begin
        origin = [100000.0, 4000000.0]
        cell_size = [30.0, -30.0]
        ndims = [100, 100]
        fidelity = 0
        dates = [Date(2020, 1, 1), Date(2020, 1, 2)]
        
        # Test constructor
        geo_data = STARSInstrumentGeoData(origin, cell_size, ndims, fidelity, dates)
        
        @test geo_data.origin == origin
        @test geo_data.cell_size == cell_size
        @test geo_data.ndims == ndims
        @test geo_data.fidelity == 0
        @test length(geo_data.dates) == 2
        
        # Test with different fidelity
        @test STARSInstrumentGeoData(origin, cell_size, ndims, 1, dates).fidelity == 1
        @test STARSInstrumentGeoData(origin, cell_size, ndims, 2, dates).fidelity == 2
    end
end
