def rayBoxIntersection(origin, direction, vmin, vmax):
        
    if direction[0] >= 0:
        tmin = (vmin[0] - origin[0]) / direction[0]
        tmax = (vmax[0] - origin[0]) / direction[0]
    else:
        tmin = (vmax[0] - origin[0]) / direction[0]
        tmax = (vmin[0] - origin[0]) / direction[0]
        
    if direction[1] >= 0:
        tymin = (vmin[1] - origin[1]) / direction[1]
        tymax = (vmax[1] - origin[1]) / direction[1]
    else:
        tymin = (vmax[1] - origin[1]) / direction[1]
        tymax = (vmin[1] - origin[1]) / direction[1]
    
    if ((tmin > tymax) or (tymin > tmax)):
        flag = 0
        tmin = -1
        return flag, tmin
    
    if (tymin > tmin):
        tmin = tymin
    
    if (tymax < tmax):
        tmax = tymax
    
    if direction[2] >= 0:
        tzmin = (vmin[2] - origin[2]) / direction[2]
        tzmax = (vmax[2] - origin[2]) / direction[2]
    else:
        tzmin = (vmax[2] - origin[2]) / direction[2]
        tzmax = (vmin[2] - origin[2]) / direction[2]
    
    if ((tmin > tzmax) or (tzmin > tmax)):
        flag = 0
        tmin = -1
        
    if (tzmin > tmin):
        tmin = tzmin
    
    if (tzmax < tmax):
        tmax = tzmax
    
    flag = 1
        
    return flag, tmin

def increment_line(volume, origin, direction, grid3D):
    flag, tmin = rayBoxIntersection(origin, direction, grid3D.minBound, grid3D.maxBound)
    
    if flag == 1:
        if tmin < 0:
            tmin = 0
        
        start = origin + tmin * direction
        boxSize = grid3D.maxBound - grid3D.minBound
        
        x = np.floor(((start[0]-grid3D.minBound[0]) / boxSize[0]) * grid3D.nx) + 1
        y = np.floor(((start[1]-grid3D.minBound[1]) / boxSize[1]) * grid3D.ny) + 1
        z = np.floor(((start[2]-grid3D.minBound[2]) / boxSize[2]) * grid3D.nz) + 1
        
        if x == grid3D.nx+1:
            x -= 1
        if y == grid3D.ny+1:
            y -= 1
        if z == grid3D.nz+1:
            z -= 1
            
        if direction[0] >= 0:
            tVoxelX = x / grid3D.nx
            stepX = 1
        else:
            tVoxelX = (x-1) / grid3D.nx
            stepX = -1
            
        if direction[1] >= 0:
            tVoxelY = y / grid3D.ny
            stepY = 1
        else: 
            tVoxelY = (y-1) / grid3D.ny
            stepY = -1
        
        if direction[2] >= 0:
            tVoxelZ = z / grid3D.nz
            stepZ = 1
        else:
            tVoxelZ = (z-1) / grid3D.nz
            stepZ = -1
            
        voxelMaxX = grid3D.minBound[0] + tVoxelX * boxSize[0]
        voxelMaxY = grid3D.minBound[1] + tVoselY * boxSize[1]
        voxelMaxZ = grid3D.minBound[2] + tVoxelZ * boxSize[2]
        
        tMaxX = tmin + (voxelMaxX - start[0]) / direction[0]
        tMaxY = tmin + (voxelMaxY - start[1]) / direction[1]
        tMaxZ = tmin + (voxelMaxZ - start[2]) / direction[2]
        
        voxelSizeX = boxSize[0] / grid3D.nx
        voxelSizeY = boxSize[1] / grid3D.ny
        voxelSizeZ = boxSize[2] / grid3D.nz
        
        tDeltaX = voxelSizeX / np.abs(direction[0])
        tDeltaY = voxelSizeY / np.abs(direction[1])
        tDeltaZ = voxelSizeZ / np.abs(direction[2])
        
        while (x<=grid3D.nx) and (x>=1) and (y<=grid3D.ny) and (y>=1) and (z<=grid3D.nz) and (z>=1):
            volume(x, y, z) += 1
            
            if tMaxX < tMaxZ:
                x = x + stepX
                tMaxX = tmaxX + tDeltaX
            else:
                z = z + stepZ
                tMaxZ = tMaxZ + tDeltaZ
            
            if tMaxY < tMaxZ:
                y = y + stepY
                tMaxY = tMaxY + tDeltaY
            else:
                z = z + stepZ
                tMaxZ = tMaxZ + tDeltaZ
                
    return volume
            
            
            