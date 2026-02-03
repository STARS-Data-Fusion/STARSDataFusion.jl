using Test
using STARSDataFusion
using Rasters
using LinearAlgebra
using SparseArrays
using Statistics
using Dates

@testset "STARSDataFusion.jl" begin
    # Include all test files
    include("test_utilities.jl")
    include("test_data_structures.jl")
    include("test_bbox.jl")
    include("test_points.jl")
    include("test_covariance_functions.jl")
    include("test_obs_operators.jl")
    include("test_spatial_utils.jl")
    include("test_fusion_basic.jl")
end
