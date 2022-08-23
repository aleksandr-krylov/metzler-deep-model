import numpy as np
from utils import *


class ShapeGenerator:
    """
    Generator of shape's path/route describing the sequence of directions one needs to take to form the shape.
    
    Parameters
    ----------
    probability : {'uniform', 'random'}, default='uniform'
        Class of the probability distribution for all the possible directions to walk.
        By default, all the directions are equally possible to be taken.
        
    random_state : int, default=None
        Controls the randomness of drawing the next direction in the walk. Initializes the new instance of
        default_rng() random generator. Same as random seed.
    """
    def __init__(self, probability='uniform', random_state=None):
        # possible directions we can walk in space
        # u - upward, r - rightward, f - forward, d - downward, l - leftward, b - backward
        self.rng = np.random.default_rng(seed=random_state) # instantiate Random Generator for reproducibility sake
        self.directions = list('ldbruf')

        if probability == 'uniform':
            self.probabilities = np.repeat(1 / len(self.directions), len(self.directions))
        elif probability == 'random':
            self.probabilities = self.rng.integers(100, size=len(self.directions))
            self.probabilities /= np.sum(self.probabilities)
            assert np.sum(self.probabilities) == 1., "Error: probabilities don't sum up to 1"
        else:
            raise ValueError(
                f"{probability} is not defined! Please, put one of the accepted arguments for probability: ('uniform', 'random')"
            )
        
        
    def update_probabilities(self, shape, overlap_likely=False, loop_likely=False):
        """
        Update the probability distribution for all possible directions given the last step by
        distributing the probability mass released by prohibited directions
        among the ones still available for the next step.
        
        Parameters
        ----------
        shape : str
            Shape's route at this moment.
            The route is represented by a sequence of direction codes in the walking order. 
            
        overlap_likely: bool, default=False
            *** todo ***
        
        loop_likely : bool, default=False
            Indicates the possibility for the closed loop to occure in the next step.
            
        
        Returns
        -------
        self : ShapeGenerator
            Updated probability distribution vector.
        """
        # get the index of the last direction
        last_idx = self.directions.index(shape[-1])
        # calculate index of the opposite direction to the last
        last_opp_idx = last_idx + 3
        last_opp_idx = last_opp_idx if last_opp_idx < 6 else last_opp_idx - 6
        
        to_be_masked = [last_idx, last_opp_idx] # canceled directions

        if overlap_likely:
            # leave out the direction which leads to having both arms overlap by the next step
            # get the previous direction we walked before taking turn
            overlap_idx = self.directions.index(shape[-2]) + 3
            overlap_idx = overlap_idx if overlap_idx < 6 else overlap_idx - 6

            to_be_masked += [overlap_idx]
        
        if loop_likely:
            # need to update probabilities one more time
            # to rule out the loop to pop up after the next step
            
            # get the index of the previous direction coming before the last one
            #prev_idx = self.directions.index(np.unique(shape))
            # calculate index of the opposite direction - the one we're not alowed to walk
            #prev_opp_idx = prev_idx + 3
            #prev_opp_idx = prev_opp_idx if prev_opp_idx < 6 else prev_opp_idx - 6

            # count number of upward steps
            ups = shape.count("u") + 1
            # count number of backward steps
            backwards = shape.count("b") + 1
            # count number of downward steps
            downs = shape.count("d") + 1
            # count number of the rest of steps left
            forwards = 10 - (ups + downs + backwards - 3)

            if ((downs < ups) and (forwards >= backwards)) or ((downs == ups) and (forwards > backwards)):
                to_be_masked += [self.directions.index("f")]

            
        # create a mask to keep track of prohibited and accessible directions to walk
        mask = np.ones_like(self.probabilities, dtype=np.bool)
        mask[np.unique(to_be_masked)] = False

        # set probs of two/three prohibited directions to zero
        self.probabilities[~mask] = 0.
        # recalculate the probability mass
        self.probabilities[mask] += (1 - np.sum(self.probabilities[mask])) / np.sum(mask)
        assert np.sum(self.probabilities) == 1., "Error: probabilities don't sum up to 1"          
    
    
    def reset_probabilities(self):
        """
        Set probabilities to default values, i.e. the ones defined at generator's init time
        
        Returns
        -------
        self : ShapeGenerator
            Default probability distribution over directions.
        """
        self.probabilities = np.repeat(1 / len(self.directions), len(self.directions))
        
        
    def draw_direction(self):
        """
        Draw the next direction to walk with respect to the probability distribution over all possible directions
        
        Returns
        -------
        d : str
            Direction's character code for the next step.
        """
        return self.rng.choice(self.directions, size=1, replace=False, p=self.probabilities)[0]
    
    
    def check_for_possible_loop(self, shape):
        """
        Scans shape's path for the possible loop at the next step.
        The loop is likely to occur if we have been walking in a plane defined by two orthogonal directions.
        
        We need to look out for the loop after 3rd step/hop by comparing the directions taken at first and last steps.
        If these two directions come to be opposite, e.g. 'l' and 'r', there is a chance to enter the loop next time.
        
        Parameters
        ----------
        shape : str
            Shape's route at this moment.
            The route is represented by a sequence of direction codes in the walking order.
            
        Returns
        -------   
        b : bool
            Boolean indication for the possibility of a loop at the next step.
        """
        d_start, d_end = shape[0], shape[-1] # directions at t=1 and t=3
        return abs(self.directions.index(d_start) - self.directions.index(d_end)) == 3


    def check_for_overalap(self, bend_point_1, bend_point_2):
        """
        """
        return (bend_point_2 - bend_point_1) == 1
            

    def generate(self, n_arms=4, step_size=2):
        """

        Generates the shape's path/route by walking in 3D space and
        iteratevely updating the distribution over possible directions to take at each step.
        
        Parameters
        ----------
        n_arms : int, default=4
            Number of arms the shape should have
            
        step_size : int, default=2
            How much we need to walk along the direction at each step.
            
            
        Returns
        -------
        path : str
            The seqeunce of characters outlining the path/route one needs to walk to form the arm-like shape.
            The path length is calculated by the following rule: len = n_hops x step_size + 1
        """
        path = ""
        overlap_likely = False
        loop_likely = False

        # make 1st arm
        to_bend_one = self.rng.integers(1, 6)
        to_bend_two = self.rng.integers(to_bend_one + 2, 8)
        to_bend_three = self.rng.integers(to_bend_two + 1, 9)

        d = "u" # always start off with walking in the up direction
        for t in range(9):
            if t == to_bend_one:
                d = "b"

            if t == to_bend_two:
                # reset all the probabilities to default values 
                self.reset_probabilities()
                # update the probability distribution of possible directions to go next
                # we need to mask the last direction drew and the one opposite to it
                # and distribute the probability mass among the rest of the available directions
                self.update_probabilities(path, overlap_likely, loop_likely)
                d = self.draw_direction()

            if t == to_bend_three:
                # check 1
                overlap_likely = self.check_for_overalap(to_bend_two, to_bend_three)
                # check 2
                loop_likely = self.check_for_possible_loop(path)

                self.reset_probabilities()
                self.update_probabilities(path, overlap_likely, loop_likely)

                d = self.draw_direction()

            path += d

        """     
        for t in range(1, n_arms + 1):
            # draw randomly a direction to start walking
            d = self.draw_direction()
            
            # walk along this direction
            path += d * step_size
            
            if t == 2: # 1st elbow is formed
                # keep going one more step to start a new elbow
                path += d
            
            # check for loops in the walk
            # we need to find out if we have been walking in one plane this far
            # loop is likely to occur at 3rd time step
            if t == 3: loop_likely = self.check_for_possible_loop(path)
                
            # reset all the probabilities to default values 
            self.reset_probabilities()
            # update the probability distribution of possible directions to go next
            # we need to mask the last direction drew and the one opposite to it
            # and distribute the probability mass among the rest of the available directions
            self.update_probabilities(path, loop_likely)
        """
                
                
        return path


class Cuboid:
    """
    Box/cube geometry class.

    Generates three-dimensional box-like shapes. A cube has 8 vertices, 12 edges and 6 faces.

    Parameters
    ----------
    x : float, default=0.
        x-coordinate of cube's center

    y : float, default=0.
        y-coordinate of cube's center

    z : float, default=0.
        z-coordinate of cube's center

    width : float, default=2.
        cube's width

    height : float, default=2.
        cube's height

    depth : float, default=2.
        cube's depth
    """
    def __init__(self, x=0., y=0., z=0., width=2., height=2., depth=2.):
        self.xc = x
        self.yc = y
        self.zc = z
        self.w = width
        self.h = height
        self.d = depth
        
        self.vertices = np.array([
            # bottom face
            (x - width/2, y - height/2, z + depth/2),
            (x - width/2, y - height/2, z - depth/2),
            (x + width/2, y - height/2, z - depth/2),
            (x + width/2, y - height/2, z + depth/2),
            # top face
            (x - width/2, y + height/2, z + depth/2),
            (x - width/2, y + height/2, z - depth/2),
            (x + width/2, y + height/2, z - depth/2),
            (x + width/2, y + height/2, z + depth/2),
        ]).T

           
        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (3, 7),
            (2, 6)
        ]
        
        self.faces = [
            # start from bottom left vertex and move clock-wise
            (0, 1, 2, 3), # bottom
            (4, 5, 6, 7), # top
            (0, 4, 7, 3), # front
            (1, 5, 6, 2), # back
            (0, 1, 5, 4), # left
            (3, 2, 6, 7), # right
        ]


    @property
    def com(self):
        """Coordinates of cube's center"""
        return np.mean(self.vertices, axis=1)
        
        
class MetzlerShape:
    """
    Generates coordinates of vertices for Metzler shape from its description.

    Metzler shape is composed of ten solid cubes attached face-to-face
    to form a rigid armlike structure with exactly three right-angled "elbows".

    Parameters
    ----------
    path : str of length 9
        Sequence of direction codes outlining the 3D shape
    """
    def __init__(self, path):
        self.directions = list('ldbruf')
        
        self.centers = [
            [0, 0, 0] # always start off with the first cube located at the origin
        ]
        # calculate centroid coordinates of cubes forming the shape
        for t, step in enumerate(path):
            # identify which direction and orientation we need to walk
            d, i = 2 * (self.directions.index(step) // 3) - 1, self.directions.index(step) % 3
            # add the coordinates for the next cube by copying the last existing cube coordinates
            self.centers += [list(self.centers[t])]
            # adjust the coordinates by following the path
            self.centers[t+1][i] += 2*d
            
        
        # compute vertices of each cube in the shape
        self.cubes = [Cuboid(*center) for center in self.centers]
        self.vertices = np.hstack([cube.vertices for cube in self.cubes])
        assert self.vertices.shape == (self.cubes[0].vertices.shape[0], len(self.cubes) * self.cubes[0].vertices.shape[1]), \
             f"Error: incorrect shape for the vertex data, {self.vertices.shape}!"

        # position the wireframe in such a way that
        # its center of mass (COM) is at the origin of its local coordinate system
        self.vertices = translate(homogenize(self.vertices), *(-1 * np.mean(self.centers, axis=0)))[:-1, :]
        # update shape's COM coordinates
        self.com = np.mean(self.vertices, axis=1)
        assert np.allclose(
            self.com,
            np.zeros_like(self.com)
        ), "Error: shape's center of mass is not at the origin (0, 0, 0)!"
        
        

        self.edges = []
        for cnt, cube in enumerate(self.cubes): self.edges += (np.array(cube.edges) + 8*cnt).tolist()
    
        self.faces = []
        for cnt, cube in enumerate(self.cubes): self.faces += (np.array(cube.faces) + 8*cnt).tolist()