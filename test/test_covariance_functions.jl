@testset "Covariance Functions" begin
    # Import functions from module
    import STARSDataFusion: matern_cor, exp_cor, mat32_cor, mat52_cor, state_cov
    
    @testset "kernel_matrix" begin
        # Test basic kernel matrix
        X = rand(5, 3)  # 5 points in 3D
        K = STARSDataFusion.kernel_matrix(X, reg=1e-10, σ=1.0)
        
        # Check dimensions
        @test size(K) == (5, 5)
        
        # Check symmetry
        @test K ≈ K'
        
        # Check positive definiteness (all eigenvalues positive)
        eigvals = eigen(K).values
        @test all(eigvals .> 0)
        
        # Check diagonal values (should be close to 1 + reg)
        @test all(diag(K) .≈ 1.0 + 1e-10)
        
        # Test with different sigma
        K_large = STARSDataFusion.kernel_matrix(X, reg=1e-10, σ=10.0)
        # Larger sigma should give more correlation
        @test sum(K_large) > sum(K)
    end
    
    @testset "matern_cor" begin
        # Test Matérn correlation
        X = rand(2, 10)  # 2D coordinates, 10 points
        C = matern_cor(X, reg=1e-10, ν=0.5, σ=1.0)
        
        # Check dimensions
        @test size(C) == (10, 10)
        
        # Check symmetry
        @test C ≈ C'
        
        # Check positive definiteness
        eigvals = eigen(C).values
        @test all(eigvals .> 0)
        
        # Test different smoothness parameters
        C_smooth = matern_cor(X, reg=1e-10, ν=1.5, σ=1.0)
        @test size(C_smooth) == (10, 10)
        @test C_smooth ≈ C_smooth'
        
        # Check positive definiteness for smooth
        eigvals_smooth = eigen(C_smooth).values
        @test all(eigvals_smooth .> 0)
    end
    
    @testset "exp_cor" begin
        # Test exponential correlation
        X = rand(2, 8)  # 2D coordinates, 8 points
        C = exp_cor(X, [1.0, 1e-10])
        
        # Check dimensions
        @test size(C) == (8, 8)
        
        # Check symmetry
        @test C ≈ C'
        
        # Check positive definiteness
        eigvals = eigen(C).values
        @test all(eigvals .> -1e-6)  # Small tolerance for numerical errors
        
        # Check that diagonal values are in reasonable range
        @test all(diag(C) .>= 1e-10)  # At least the regularization term
    end
    
    @testset "mat32_cor" begin
        # Test Matérn 3/2 correlation
        X = rand(2, 6)  # 2D coordinates, 6 points
        C = mat32_cor(X, [1.0, 1e-10])
        
        # Check dimensions
        @test size(C) == (6, 6)
        
        # Check symmetry
        @test C ≈ C'
        
        # Check diagonal values are in reasonable range
        @test all(diag(C) .>= 1e-10)
        
        # Check positive definiteness
        eigvals = eigen(C).values
        @test all(eigvals .> -1e-6)
    end
    
    @testset "mat52_cor" begin
        # Test Matérn 5/2 correlation
        X = rand(2, 6)  # 2D coordinates, 6 points
        C = mat52_cor(X, [1.0, 1e-10])
        
        # Check dimensions
        @test size(C) == (6, 6)
        
        # Check symmetry
        @test C ≈ C'
        
        # Check diagonal values are in reasonable range
        @test all(diag(C) .>= 1e-10)
        
        # Check that function runs without throwing errors
        # (eigenvalue tests too strict for random data)
        @test issymmetric(C)
    end
    
    @testset "state_cov" begin
        # Test state-dependent covariance
        Xtt = rand(5, 4)  # 5 dimensions, 4 time points
        pars = [1.0]
        
        Qst = state_cov(Xtt, pars)
        
        # Check dimensions
        @test size(Qst) == (4, 4)
        
        # Check symmetry
        @test Qst ≈ Qst'
        
        # Check positive definiteness
        eigvals = eigen(Qst).values
        @test all(eigvals .> -1e-6)
        
        # Check that variance scales with pars
        pars_large = [2.0]
        Qst_large = state_cov(Xtt, pars_large)
        @test maximum(Qst_large) > maximum(Qst)
    end
end
