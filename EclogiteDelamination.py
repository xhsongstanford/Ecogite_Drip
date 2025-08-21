

# This file can only run with successfully installed Underworld2 Package.

# I cannot find and install that package in when building this Capsule

# You can view the code structure and an example output here, but it is not runnable online.


import underworld as uw
from underworld import function as fn
import underworld.visualisation as vis
import math
import numpy as np
import os
import pathlib

rank = uw.mpi.rank

# Define the model area
boxLength      = 8.0
boxHeight      = 4.0
ref_viscosity  = 1.0

# Set the vsicosities
LithoMantleViscosity =   0.01
LowerMantleViscosity =   0.1
CrustViscosity       =   50.0
EclogiteViscosity    =    0.5

outputEvery  = 4

outputPath = '../results/'

old_files = pathlib.Path(outputPath).glob('*')

# numerical parameters - demo or 'longtest'
# demo       - settings for a quick run (iff longtest is False)
# 'longtest' - settings for benchmark

longtest = False

model_end_time   = 200.
res              = 16
stokes_inner_tol = 1e-7
stokes_outer_tol = 1e-6
# users ignore
import os
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
for path in old_files:
    path.unlink()

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (8*res, 4*res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight))

velocityField = mesh.add_variable(         nodeDofCount=2 )
pressureField = mesh.subMesh.add_variable( nodeDofCount=1 )


# Create a swarm.
swarm = uw.swarm.Swarm( mesh=mesh )

# Create a data variable. It will be used to store the material index of each particle.
materialIndex = swarm.add_variable( dataType="int", count=1 )

# Create a layout object, populate the swarm with particles.
swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarm.populate_using_layout( layout=swarmLayout )


# define these for convience. 

from underworld.function.math import exp

CrustIndex = 0
EclogiteIndex = 1
LithoMantleIndex = 2
LowerMantleIndex = 3

# material perturbation from van Keken et al. 1997
# lowerMantleY   = 2.5
CrustShape  = np.array([ (0.0, 4.0), (0.0, 3.5), (2.3, 3.5), (2.8, 3.65), (5.2, 3.65), (5.7, 3.5), (8.0, 3.5), (8.0, 4.0) ])
EclogiteShape = np.array([ (1.3, 4.0), (1.3, 3.55), (2.1, 3.2), (4.0, 3.3), (4.4, 3.4), (4.4, 4.0) ])
#LithoMantleShape = np.array([ (0.0, 4.0), (0.0, 2.6), (8.0, 2.6), (8.0, 4.0) ])
LithoMantleShape = np.array([ (0.0, 4.0), (0.0, 2.6), (1.7, 2.6), (3.2, 2.5), (4.7, 2.6), (8.0, 2.6), (8.0, 4.0) ])

Crust = fn.shape.Polygon( CrustShape )
Eclogite  = fn.shape.Polygon( EclogiteShape )
LithoMantle  = fn.shape.Polygon( LithoMantleShape )


# Create function to return particle's coordinate
coord = fn.coord()

#xx = coord[0]
#curve = -(xx - 1.9)/(1. + 1.*exp(8.*(xx - 2.2)))
#curve2 = -(xx - 1.1)/(1. + 1.*exp(3.*(xx - 1.0)))
#curve3 = -(xx - 1.0)/(1. + 1.*exp(2.*(xx - 1.0)))

# Define the material perturbation.
sigma_e = 0.7
sigma_l = 1.5
Eclogite = 3.5 + 0.2 * fn.math.exp(-1.7**2/(2*sigma_e**2)) - 0.2*fn.math.exp(-(coord[0]-4.0)**2/(2*sigma_e**2))   #fn.math.cos( k*coord[0] )
LithoMantle = 2.6 - 0.2*fn.math.exp(-(coord[0]-4.0)**2/(2*sigma_l**2))
# perturbationFn = offset + amplitude*curve - 0.8*amplitude*curve2  #fn.math.cos( k*coord[0] )

# Setup the conditions list. 
# If z is less than the perturbation, set to lightIndex.
conditions = [ ( Eclogite < coord[1] , EclogiteIndex ),
               ( LithoMantle < coord[1] , LithoMantleIndex ),
               (                   True , LowerMantleIndex ) ]

# The swarm is passed as an argument to the evaluation, providing evaluation on each particle.
# Results are written to the materialIndex swarm variable.
materialIndex.data[:] = fn.branching.conditional( conditions ).evaluate(swarm)

# initialise everying to be upper mantle material
# materialIndex.data[:] = LowerMantleIndex

# change matieral index if the particle is not upper mantle
for index in range( len(swarm.particleCoordinates.data) ):
    coord = swarm.particleCoordinates.data[index][:]
    if Crust.evaluate(tuple(coord)):
        materialIndex.data[index] = CrustIndex
    #elif Eclogite.evaluate(tuple(coord)):
    #elif Eclogite > coord[1]:
    #    materialIndex.data[index] = EclogiteIndex
    #elif LithoMantle.evaluate(tuple(coord)):
    #elif LithoMantle > coord[1]:
    #    materialIndex.data[index] = LithoMantleIndex

# initialise 
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

store = vis.Store('output/subduction')

# plot the model
fig1 = vis.Figure(figsize=(800,400), quality=2, rulers=True)
fig1.append( vis.objects.Points(swarm, materialIndex, pointSize=2, colourBar=False) )
fig1.append( vis.objects.VectorArrows( mesh, velocityField))


viscosityMap = { CrustIndex : CrustViscosity, 
                 EclogiteIndex    : EclogiteViscosity, 
                 LithoMantleIndex : LithoMantleViscosity,
                 LowerMantleIndex : LowerMantleViscosity}
viscosityMapFn = fn.branching.map( fn_key = materialIndex, mapping = viscosityMap )

# LowerMantleDensity = 0.4
# LithoMantleDensity = 0.45
# EclogiteDensity    = 0.60
# CrustDensity       = 0.0

LowerMantleDensity = 3.20
LithoMantleDensity = 3.25
EclogiteDensity    = 3.40
CrustDensity       = 2.80

densityMap = {   CrustIndex       : CrustDensity, 
                 EclogiteIndex    : EclogiteDensity, 
                 LithoMantleIndex : LithoMantleDensity,
                 LowerMantleIndex : LowerMantleDensity}

# Set a density of '0.' for light material, '1.' for dense material.
# densityMap   = { lightIndex:0., denseIndex:0.1 }
densityFn    = fn.branching.map( fn_key = materialIndex, mapping = densityMap )

# Set a viscosity value of '1.' for both materials.
# viscosityMap = { lightIndex: ref_viscosity, denseIndex: viscosityRatio*ref_viscosity }
fn_viscosity  = fn.branching.map( fn_key = materialIndex, mapping = viscosityMap )

# Define a vertical unit vector using a python tuple.
z_hat = ( 0.0, 1.0 )

# Create buoyancy force vector
buoyancyFn = -1.0*densityFn*z_hat

# initialise 
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))
strainRate_matrix = fn.tensor.symmetric(velocityField.fn_gradient)


velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.

# Construct node sets using the mesh specialSets
iWalls = mesh.specialSets["Left_VertexSet"]   + mesh.specialSets["Right_VertexSet"]
jWalls = mesh.specialSets["Bottom_VertexSet"] + mesh.specialSets["Top_VertexSet"]
leftrightbottom = iWalls + jWalls - mesh.specialSets["Top_VertexSet"]
allWalls = iWalls + jWalls
bottomWall = mesh.specialSets["Bottom_VertexSet"]
# Prescribe degrees of freedom on each node to be considered Dirichlet conditions.
# In the x direction on allWalls flag as Dirichlet
# In the y direction on jWalls (horizontal) flag as Dirichlet
stokesBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                             indexSetsPerDof = (allWalls, jWalls) )

stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = stokesBC,
                            fn_viscosity  = fn_viscosity, 
                            fn_bodyforce  = buoyancyFn )

solver = uw.systems.Solver( stokes )

# Optional solver settings
if(uw.mpi.size==1):
    solver.set_inner_method("lu")
solver.set_inner_rtol(stokes_inner_tol) 
solver.set_outer_rtol(stokes_outer_tol) 

# Create a system to advect the swarm
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )

#Plot of Strain Rate, 2nd Invariant
figStrainRate = vis.Figure(figsize=(800,400), quality=2, rulers=True)
figStrainRate.append( vis.objects.Surface(mesh, strainRate_2ndInvariant, logScale=True) )
#Plot of particles stress invariant
figStress = vis.Figure(figsize=(800,400), quality=2, rulers=True)
figStress.append( vis.objects.Points(swarm, 2.0*viscosityMapFn*strainRate_2ndInvariant, pointSize=2, logScale=True) )

# Initialise time and timestep.
time = 0.
step = 0

# parameters for output
timeVal     = []
vrmsVal     = []

# Save mesh and retain file handle for future xdmf creation
meshFileHandle = mesh.save(outputPath+"Mesh.h5")
swarmFileHandle = swarm.save(outputPath+"Swarm.h5")

# define an update function
def update():
    dt = advector.get_max_dt() # retrieve the maximum possible timestep from the advection system.
    print('******* dt = '+str(dt))
    advector.integrate(dt)     # advect step.
    return time+dt, step+1

model_end_time   = 20.
t_list = []
step_list = []
while time*0.323 <= model_end_time:

    # Get solution
    solver.solve()
    
    # Calculate the RMS velocity.
    vrms = stokes.velocity_rms()

    # Record values into arrays
    if(uw.mpi.rank==0):
        vrmsVal.append(vrms)
        timeVal.append(time)
    
    # Output to disk
    if step%outputEvery == 0 :#or 40 - time*0.323 < 2:
        
        t_list.append(time*0.323)
        step_list.append(step)
        #print('dt = '+str(dt))
        
        #if(uw.mpi.rank==0):
        print('step = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}'.format(step,time,vrms))

        filename = outputPath+"/velocityField."+str(step)
        vFH      = velocityField.save(filename+".h5")
        #velocityField.xdmf( filename, vFH, "velocity", meshFileHandle, "Mesh", time )
        
        filename = outputPath+"/material."+str(step)
        mID = materialIndex.evaluate(swarm)
        np.savetxt(filename+".txt", mID)

        filename = outputPath+"/materialxy." + str(step)
        mID = swarm.particleCoordinates.data
        np.savetxt(filename+".txt", mID)

        filename = outputPath+"/materialmesh." + str(step)
        mID = materialIndex.evaluate(mesh)
        np.savetxt(filename+".txt", mID)

        filename = outputPath+"/pressureField."+str(step)
        pFH      = pressureField.save(filename+".h5")
        #pressureField.xdmf(filename, pFH, "pressure", meshFileHandle, "Mesh", time )

        filename = outputPath+"/strainInv."+str(step)
        sFH = strainRate_2ndInvariant.evaluate(mesh)
        np.savetxt(filename+".txt", sFH)

        filename = outputPath+"/strainMatrix."+str(step)
        sFH = strainRate_matrix.evaluate(mesh)
        np.savetxt(filename+".txt", sFH)
        
        outputFilename = outputPath+"particle"+str(step).zfill(4)+str(round(time*0.323)).zfill(5)+'.png'
        fig1.save_image(outputFilename)
        outputFilename = outputPath+"strainrate"+str(step).zfill(4)+str(round(time*0.323)).zfill(5)+'.png'
        figStrainRate.save_image(outputFilename)
        outputFilename = outputPath+"shearstress"+str(step).zfill(4)+str(round(time*0.323)).zfill(5)+'.png'
        figStress.save_image(outputFilename)



    # We are finished with current timestep, update.
    time, step = update()


np.savetxt(outputPath+'timelist.txt', np.array(t_list))
np.savetxt(outputPath+'steplist.txt', np.array(step_list))
if(uw.mpi.rank==0):
    print('step = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}'.format(step,time,vrms))


