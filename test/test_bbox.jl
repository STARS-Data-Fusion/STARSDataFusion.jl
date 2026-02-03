@testset "BBox Tests" begin
    using STARSDataFusion.BBoxes
    
    @testset "BBox Constructor" begin
        # Test basic constructor
        bbox = BBox(0.0, 0.0, 10.0, 10.0)
        @test bbox.xmin == 0.0
        @test bbox.ymin == 0.0
        @test bbox.xmax == 10.0
        @test bbox.ymax == 10.0
        
        # Test constructor from Raster (2D)
        raster_2d = Raster(rand(10, 10), dims=(X(0.0:1.0:9.0), Y(0.0:1.0:9.0)))
        bbox_from_raster = BBox(raster_2d)
        @test bbox_from_raster.xmin < bbox_from_raster.xmax
        @test bbox_from_raster.ymin < bbox_from_raster.ymax
        
        # Test constructor from Raster (3D)
        raster_3d = Raster(rand(10, 10, 3), dims=(X(0.0:1.0:9.0), Y(0.0:1.0:9.0), Ti(1:3)))
        bbox_from_3d = BBox(raster_3d)
        @test bbox_from_3d.xmin < bbox_from_3d.xmax
        @test bbox_from_3d.ymin < bbox_from_3d.ymax
    end
    
    @testset "buffer Function" begin
        bbox = BBox(0.0, 0.0, 10.0, 10.0)
        
        # Test positive buffer
        buffered = buffer(bbox, 5.0)
        @test buffered.xmin == -5.0
        @test buffered.ymin == -5.0
        @test buffered.xmax == 15.0
        @test buffered.ymax == 15.0
        
        # Test negative buffer (shrinking)
        shrunk = buffer(bbox, -2.0)
        @test shrunk.xmin == 2.0
        @test shrunk.ymin == 2.0
        @test shrunk.xmax == 8.0
        @test shrunk.ymax == 8.0
        
        # Test zero buffer
        unchanged = buffer(bbox, 0.0)
        @test unchanged.xmin == bbox.xmin
        @test unchanged.ymin == bbox.ymin
        @test unchanged.xmax == bbox.xmax
        @test unchanged.ymax == bbox.ymax
    end
    
    @testset "calculate_centroid Function" begin
        # Test centroid of unit square
        bbox = BBox(0.0, 0.0, 10.0, 10.0)
        centroid = calculate_centroid(bbox)
        @test centroid.x == 5.0
        @test centroid.y == 5.0
        
        # Test centroid of offset box
        bbox2 = BBox(-10.0, -5.0, 10.0, 5.0)
        centroid2 = calculate_centroid(bbox2)
        @test centroid2.x == 0.0
        @test centroid2.y == 0.0
        
        # Test centroid of non-square box
        bbox3 = BBox(0.0, 0.0, 20.0, 10.0)
        centroid3 = calculate_centroid(bbox3)
        @test centroid3.x == 10.0
        @test centroid3.y == 5.0
    end
end
