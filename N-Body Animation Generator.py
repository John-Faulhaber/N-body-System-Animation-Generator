########################
### Preliminary code ###
########################


import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')
G = 6.67408e-11 # gravitational constant in mks
au = 149597870700.0 # meters in one AU
Msun = 1.989e30 # mass of the sun in mks
pc = 3.085677581e+16 # meters in one parsec
day = 24.0*60.0*60.0 # seconds in one day


#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################


########################################
### defining the necessary functions ###
########################################


def forceMagnitude(mi, mj, sep):
    """
    Compute magnitude of gravitational force between two particles.
    
    Parameters
    ----------
    mi, mj : float
        Particle masses in kg.
    sep : float
        Particle separation (distance between particles) in m.
        
    Returns
    -------
    force : float
        Gravitational force between particles in N.
    
    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            print magnitudeOfForce(mEarth, mPerson, radiusEarth)
        Output:
            683.935546875
    """
    G = 6.67e-11                # m3 kg-1 s-2
    return G * mi * mj / sep**2 # N


def magnitude(vec):
    """
    Compute magnitude of any vector with an arbitrary number of elements.
    
    Parameters
    ----------
    vec : numpy array
        Any vector
        
    Returns
    -------
    magnitude : float
        The magnitude of that vector.
        
    Example
    -------
        Input:
            print magnitude(np.array([3.0, 4.0, 0.0]))
        Output:
            5.0    
    """
    return np.sqrt(np.sum(vec**2))


def unitDirectionVector(pos_a, pos_b):
    """
    Create unit direction vector from pos_a to pos_b
    
    Parameters
    ----------
    pos_a, pos_b : two numpy arrays
        Any two vectors
        
    Returns
    -------
    unit direction vector : one numpy array (same size input vectors)
        The unit direction vector from pos_a toward pos_b
        
    Example
    -------
        Input:
            someplace = np.array([3.0,2.0,5.0])
            someplaceelse = np.array([1.0, -4.0, 8.0])
            print unitDirectionVector(someplace, someplaceelse)
        Output:
            [-0.28571429, -0.85714286,  0.42857143]
    """
    
    # calculate the separation between the two vectors
    separation = pos_b - pos_a
    
    # divide vector components by vector magnitude to make unit vector
    return separation/magnitude(separation)


def forceVector(mi, mj, pos_i, pos_j):
    """
    Compute gravitational force vector exerted on particle i by particle j.
    
    Parameters
    ----------
    mi, mj : floats
        Particle masses, in kg.
    pos_i, pos_j : numpy arrays
        Particle positions in cartesian coordinates, in m.
        
    Returns
    -------
    forceVec : numpy array
        Components of gravitational force vector, in N.
        
    Example
    -------
        Input:
            mEarth = 6.0e24     # kg
            mPerson = 70.0      # kg
            radiusEarth = 6.4e6 # m
            centerEarth = np.array([0,0,0])
            surfaceEarth = np.array([0,0,1])*radiusEarth
            print forceVector(mEarth, mPerson, centerEarth, surfaceEarth)
            
        Output:
            [   0.            0.          683.93554688]
    
    
    """
    
    # compute the magnitude of the distance between positions
    distance = magnitude(pos_i - pos_j) 
    # this distance is in meters, because pos_i and pos_j were.
    
    # compute the magnitude of the force
    force = forceMagnitude(mi, mj, distance)
    # the magnitude of the force is in Newtons
    
    # calculate the unit direction vector of the force
    direction = unitDirectionVector(pos_i, pos_j) 
    # this vector is unitless, its magnitude should be 1.0
    
    return force*direction # a numpy array, with units of Newtons


def calculateForceVectors(masses, positions):
    
    au = 1.496e11
    # how many particles are there?
    N = len(positions)

    # create an empty list, which we will fill with force vectors
    forcevectors = []

    #plotParticles(masses, positions)    
    
    # loop over particles for which we want the force vector
    for i in range(N):

        # create a force vector with all three elements as zero
        vector = np.zeros(3)

        # loop over all the particles we need to include in the force sum
        for j in range(N):

            # as long as i and j are not the same...
            if j != i:

                # ...add in the force vector of particle j acting on particle i
                vector += forceVector(masses[i], masses[j], positions[i], positions[j])

        # append this force vector into the list of force vectors
        forcevectors.append(vector)


    # return the list of force vectors out of the function
    return forcevectors


def updateParticles(masses, positions, velocities, dt):
    """
    Evolve particles in time via leap-frog integrator scheme. This function
    takes masses, positions, velocities, and a time step dt as

    Parameters
    ----------
    masses : np.ndarray
        1-D array containing masses for all particles, in kg
        It has length N, where N is the number of particles.
    positions : np.ndarray
        2-D array containing (x, y, z) positions for all particles.
        Shape is (N, 3) where N is the number of particles.
    velocities : np.ndarray
        2-D array containing (x, y, z) velocities for all particles.
        Shape is (N, 3) where N is the number of particles.
    dt : float
        Evolve system for time dt (in seconds).

    Returns
    -------
    Updated particle positions and particle velocities, each being a 2-D
    array with shape (N, 3), where N is the number of particles.

    """

    startingPositions = np.array(positions)
    startingVelocities = np.array(velocities)

    # how many particles are there?
    nParticles, nDimensions = startingPositions.shape

    # make sure the three input arrays have consistent shapes
    assert(startingVelocities.shape == startingPositions.shape)
    assert(len(masses) == nParticles)

    # calculate net force vectors on all particles, at the starting position
    startingForces = np.array(calculateForceVectors(masses, startingPositions))

    # calculate the acceleration due to gravity, at the starting position
    startingAccelerations = startingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending position
    nudge = startingVelocities*dt + 0.5*startingAccelerations*dt**2
    endingPositions = startingPositions + nudge

    # calculate net force vectors on all particles, at the ending position
    endingForces = np.array(calculateForceVectors(masses, endingPositions))

    # calculate the acceleration due to gravity, at the ending position
    endingAccelerations = endingForces/np.array(masses).reshape(nParticles, 1)

    # calculate the ending velocity
    endingVelocities = (startingVelocities +
                        0.5*(endingAccelerations + startingAccelerations)*dt)

    return endingPositions, endingVelocities


def tinyCluster(N=20, maximum_mass=0.01*Msun):
    '''This function creates N-body initial conditions for
    a (very) cartoon model of stellar cluster.

    WARNING: With these initial conditions, it's very easy
    for some of your particles to have very close approaches.
    This means, to properly resolve their motion, you either
    need to:

        (a) take very short time steps so you accurately
        capture the accelerations of these close approaches

        (b) modify your force of gravity calculation by
        including a "softening length". That is, in the
        place where you calculate F = GMm/r**2, you instead
        calculate the magnitude of the force as GMm/s**(2,
        where s = np.sqrt(r**2 + epsilon**2) where epsilon
        is some small number like 0.1 AU. This "softens" the
        strong forces that would otherwise result from very
        close approaches.

    Inputs:
        N (= 30 by default)
            the total number of particles to create
        maximum_mass (= 0.01 solar masses by default)
            the maximum mass of the particles that can go
            into the cluster; the masses of the particles
            will be random, drawn from a uniform distribution
            with this as the maximum

    Outputs:
        this function returns three arrays
            masses [shape is (nParticles)]
            positions [shape (nParticles, nDimensions)]
            velocities [shape (nParticles, nDimensions)]

    Example Usage:
        mParticles, initialPositions, initialVelocities = tinyCluster()
    '''

    # set up the masses
    masses = np.random.uniform(0, 1, N)*maximum_mass

    # convert to cartesian coordinates
    positions = np.random.normal(0, 1.0, [N,3])*au
    radii = np.sqrt(np.sum(positions**2, 1))
    mass_enclosed = np.array([np.sum(masses[radii <= r]) for r in radii])
    sigma = np.sqrt(G*mass_enclosed/radii)

    #directions = np.array([np.cross(np.random.uniform(size=3), positions[i]) for i in range(N)])
    #for i in range(N):
    #    directions[i,:] /= np.sqrt(np.sum(directions[i,:]**2))

    # calculate velocities for circular orbits
    #velocities = (sigma*np.random.normal(0,1,N))[:,np.newaxis]*directions
    velocities = sigma[:,np.newaxis]*np.random.normal(0,1,[N,3])*0.5

    # return them as three separate arrays
    return masses, positions, velocities


def calculateTrajectories(masses,initialPositions,initialVelocities,totalTimeEvolveSystem,timeStepSize):
    """ 
     Calculates updated values
     
     Parameters
     ----------
         masses: An array of n masses corresponding to the n bodies in question. 
         initialPositions: An array of shape (2,3) containing the x,y, and z position data for the n bodies in question.
         initialVelocities: An array of shape (2,3) containing the x,y, and z velocity data for the n bodies in question.
         totalTimeEvolveSystem: A time value for which to evolve the system. (in seconds)
         timeStepSize: The value for the time step at which to evaluate each iteration. (in seconds)
    
    
    returns
    -------
        An array of time values, an array of position values, and an array of velocity valuesw, all three of
        which having the same shape as their inition input arrays.
    
    
    
    example
    -------
        calculateTrajectories(testDataMass,testDataPos,testDataVel,100,10)
    
    
    
    """
    
    positions=[]
    velocities=[]
    times=np.arange(0,totalTimeEvolveSystem+timeStepSize,timeStepSize)
    for i in range(len(times)): 
        values=updateParticles(masses, initialPositions, initialVelocities,timeStepSize)
        positions.append(values[0])
        velocities.append(values[1])
        initialPositions=values[0]
        initialVelocities=values[1]
    return times,np.array(positions),np.array(velocities)


#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################


######################################
### Choosing the system parameters ###
######################################
### NOTICE: This will create a new system each time it is run. Revisit the this code and save the produced initial values manually ("mytimes", "mypositions", etc.) in a text file (careful not to re-run everything), if the produced animation is favorable, and you wish to save the initial conditions for further use (higher dpi, perhaps a longer animation, more detailed calculations, etc.).

mySystemMass,mySystemPos,mySystemVel=tinyCluster(N=3, maximum_mass=(250*Msun))
mytimes,mypositions,myvelocities=calculateTrajectories(mySystemMass,mySystemPos,mySystemVel,864000,864)


#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################


##############################
### Creating the animation ###
##############################


from mpl_toolkits import mplot3d # allows for 3D plot
import matplotlib.animation as ani # imports animation package
FFMpegWriter = ani.writers['ffmpeg']
writer = ani.FFMpegWriter(fps=60) # the resulting animation can be tempermental depending on this value. Cooperation seems to vary across monitors.
fig = plt.figure()
fileName="N-Body_Animation.mp4"
with writer.saving(fig, fileName, 400): # the number here is the dpi of the produced frames (higher dpi = higher quality, longer run time). If you do not have a very fast computer, start with a much lower value (ie. 40), and increase as your patience permits. See matplotlib.animation parameters for further options. 
    print "Starting animation creation..."
    fc=range(len(mytimes))  # frame count
    
    ############################
    ### FOR TRAILS, USE THIS ### (see next line)
    ############################
    ### WARNING: Adding trails greatly lengthens code run time.
    
    ax = plt.axes(projection='3d')
    
    
    for i in fc:
        
        # a bunch of plotting and formatting
        
        ###############################
        ### FOR NO TRAILS, USE THIS ###  (see next line)
        ###############################
        
        #ax = plt.axes(projection='3d')
        
        ### IMPORTANT ### The list of colors in the following line MUST have the same "N" amount of elements as the chosen "N" amount of particles.
        b1=ax.scatter3D(mypositions[i,:,0],mypositions[i,:,1],mypositions[i,:,2],c=['red','blue','green'],s=(mySystemMass[:]/np.max(mySystemMass))*5) # plot data for frame "i"
        ax.set_xlabel('\nx-position (m)',labelpad=7)
        ax.set_ylabel('\n\ny-position (m)',labelpad=7)
        ax.set_zlabel('\n\n\nz-position (m)',labelpad=2)
        ax.set_title(('''My "tinyCluster" System\n state at {} days''').format((mytimes[i])/(day)),y=1.1)
        ax.set_xlim(min(mypositions[0,:,0])*2, max(mypositions[0,:,0])*2) # starts with limits two times the minimum necessary to show all particles so as to keep scaling general, and to have a buffer to view interactions
        ax.set_ylim(min(mypositions[0,:,1])*2, max(mypositions[0,:,1])*2)
        ax.set_zlim(min(mypositions[0,:,2])*2, max(mypositions[0,:,2])*2)
        ax.view_init(25, -60)
        if i==0:
            for i in range(0,120): # mimics/creates a "pause" in the animation - length of "pause" is determined by the frame rate chosen, and how many frames are created for the "pause"
                writer.grab_frame() # saves a "pause" frame
            print "Done"
            print "Animating initial visual orientation..."
            for angle in range(-60, 301): # makes 3D plot spin 360 degrees
                ax.view_init(25, angle)
                plt.draw()
                writer.grab_frame() # saves a rotation frame
            print "Done"
            print "Updating positions..."
            for i in range(0,120): # mimics/creates a "pause" in the animation - length of "pause" is determined by the frame rate chosen, and how many frames are created for the "pause"
                writer.grab_frame() # saves a "pause" frame
        writer.grab_frame() # saves a frame
        if i % 100 == 0: # gives progress update every 100 frames
            print "Saved position update {}/{}.".format(i, max(fc))
    for i in range(0,60): # mimics/creates a "pause" in the animation - length of "pause" is determined by the frame rate chosen, and how many frames are created for the "pause"
        writer.grab_frame() # saves a "pause" frame 
    ax.set_title(('''My "tinyCluster" System\n state at {} days\nzoomed out''').format(10,y=1.1)) # formats and adds a title to the plot
    for i in range(0,60): # mimics/creates a "pause" in the animation - length of "pause" is determined by the frame rate chosen, and how many frames are created for the "pause"
        writer.grab_frame() # saves a "pause" frame
    ax.set_xlim(min(mypositions[-1,:,0]), max(mypositions[-1,:,0])) # zooms out (jump-cut) to show all particles (general - will allways go to what is required to see all bodies)
    ax.set_ylim(min(mypositions[-1,:,1]), max(mypositions[-1,:,1]))
    ax.set_zlim(min(mypositions[-1,:,2]), max(mypositions[-1,:,2]))
    for i in range(0,120): # mimics/creates a "pause" in the animation - length of "pause" is determined by the frame rate chosen, and how many frames are created for the "pause"
        writer.grab_frame() # saves a "pause" frame
    print "Animating final visual orientation..."
    for angle in range(-60, 301): # makes 3D plot spin 360 degrees
        ax.view_init(25, angle)
        plt.draw()
        writer.grab_frame() # saves plot rotation frames
    print "Done"
    print "Animation complete."
    import os
    print "Your animation file can be found here:\n {}".format(os.getcwd()+'/'+fileName)
