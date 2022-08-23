import math
import numpy as np
import os
from pathlib import Path
from utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle



class Camera:
    """
    Camera class represents the virtual camera used to view the scene.
    The camera itself is not rendered, but its transform affects the apparent placement of the objects
    in the rendered image of the scene.

    Parameters
    ----------
    fovy: float, default=90 (degrees in radians)
            The vertical (in y direction) field of view angle, in radians.
        
    znear: float, default=0.1
        The distance from the viewer to the near (front) clipping plane of the viewing frustum (always positive).
        
    zfar: float, default=100.
        The distance from the viewer to the far (back) clipping plane of the viewing frustum (always positive).

    """
    def __init__(self, fovy=math.radians(90), znear=0.1, zfar=100.):
        self.position = np.array([0., 0., 0.]) # by default, camera is placed at the origin
        self.fovy = fovy
        self.znear = znear
        self.zfar = zfar


    def setPosition(self, xangle, yangle, zangle, tx, ty, tz):
        """
        Set the camera position in the world space.

        Parameters
        ----------
        xangle : float
            Rotation angle around x-axis

        yangle : float
            Rotation angle around y-axis

        zangle : float
            Rotation angle around z-axis

        tx : float
            How much to move along x-axis

        ty : float
            How much to move along y-axis

        tz : float
            How much to move along z-axis

    
        Returns
        -------
        Updated camera position
        """
        transform = np.identity(4)
        transform = translate(transform, tx, ty, tz)
        transform = zrotate(transform, math.radians(zangle))
        transform = yrotate(transform, math.radians(yangle))
        transform = xrotate(transform, math.radians(xangle))
        # the final position of the camera can be extracted from
        # the entries of the last column of the transform matrix
        self.position = transform[:-1, -1]


    def setSphericalPosition(self, r, theta, phi):
        """
        """
        elevation = math.radians(theta)
        azimuth = math.radians(phi)
        radius = np.array([0., 0., r])

        rotation = yrotation(azimuth) @ xrotation(elevation)
        position = rotation @ homogenize(radius)

        self.position = position[:-1]
        


    def setLookAtMatrix(self, at=np.array([0., 0., 0.]), up=np.array([0., 1., 0.])):
        """
        Set up LookAt matrix defining the transformation required to view the object
        """
        return lookat(eye=self.position, at=at, up=up)

    
    def setProjectionMatrix(self, aspect=1.):
        """
        Set up perspective projection matrix
        """
        return perspective(self.fovy, aspect, self.znear, self.zfar)


class Object3D:
    """
    """
    def __init__(self, shape, vertexcolor=None, vertexsize=None, edgecolor='black', edgewidth=1.5, facecolor=None):
        self.vertices = shape.vertices
        self.edges = shape.edges
        self.faces = shape.faces
        self.com = shape.com
        self.attributes = {
            'vertexcolor': vertexcolor,
            'vertexsize': vertexsize,
            'edgecolor': edgecolor,
            'edgewidth': edgewidth,
            'facecolor': facecolor
        }

    
    @property
    def n_vertices(self):
        return len(self.vertices)


    @property
    def n_edges(self):
        return len(self.edges)


    @property
    def n_faces(self):
        return len(self.faces)


    def setModelMatrix(self):
        """
        Set up model matrix defining transformation from the model space to the world space

        By default, the object is placed at the origin of world's coordinate system, i.e.
        object's origin coincides with the world's.
        """
        return np.identity(4)



class VertexShader:
    """
    """
    def __init__(self):
        self.zbuffer = [] # depth buffer


    def transform(self, mvp, object):
        """
        """
        # project vertex 3D positions from the camera space to the 2D coordinates in the screen space
        projected = mvp @ homogenize(object.vertices) # now vertices are defined in the clip space
        assert projected.shape == (4, object.vertices.shape[1]), "Error: projection is done incorrectly!"

        # keep the depth information in Z-buffer for visibility determination in further rendering
        self.zbuffer = [projected[-1, face] for face in object.faces]
        # normalize the coordinates by doing perspective divide
        normalized = [
            projected[:, face] / self.zbuffer[idx] for idx, face in enumerate(object.faces)
        ] # now vertices are defined in NDC space [-1, 1]

        return [face_vertices[:2, :] for face_vertices in normalized] # screen coordinates X, Y


class GeometryShader:
    """
    """
    def __init__(self, geometry=Polygon):
        self.primitive = geometry


    def generate(self, vertices, zbuffer, properties):
        """
        """
        kwargs = {
            "fill" : True if properties.get("facecolor") else False,
            "facecolor" : properties.get("facecolor"),
            "edgecolor" : properties.get("edgecolor"),
            "linewidth" : properties.get("edgewidth"),
        }
        
        collection = [] # a list of generated primitives given the vertex data
        for cnt, face in enumerate(vertices):
            collection.append(
                self.primitive(xy=face.T, zorder=-np.mean(zbuffer[cnt]), **kwargs)
            )

        return collection


class Renderer:
    """
    """
    def __init__(self, imgsize=(128, 128), dpi=100, bgcolor="white", format="png"):
        self.vbuffer = [] # stores vertex data

        # configure matplotlib properties and styles 
        mpl.rcParams["figure.figsize"] = (imgsize[0] / dpi, imgsize[1] / dpi) 	# figure size in inches
        mpl.rcParams["figure.dpi"] = dpi 							            # figure dots per inch
        mpl.rcParams["figure.autolayout"] = True 			                    # when True, automatically adjust subplot
                                                                                # parameters to make the plot fit the figure
                                                                                # using `tight_layout`
        mpl.rcParams["figure.facecolor"] = bgcolor
        mpl.rcParams["figure.edgecolor"] = bgcolor

        mpl.rcParams["axes.facecolor"] = bgcolor
        mpl.rcParams["axes.edgecolor"] = bgcolor

        mpl.rcParams["savefig.dpi"] = dpi						                # figure dots per inch or 'figure'
        mpl.rcParams["savefig.format"] = format 				                # figure format {png, ps, pdf, svg} when saving
        mpl.rcParams["savefig.facecolor"] = bgcolor 		                    # figure face color when saving
        mpl.rcParams["savefig.edgecolor"] = bgcolor 		                    # figure edge color when saving
        
        plt.close("all") # close all the figures created by matplotlib before 
        # create a figure to render the object in it
        self.figure = plt.figure(
                tight_layout={"pad": 0.0, "w_pad": None, "h_pad" : None, "rect": None}
        )

    
    def render(self, object, camera):
        """
        """
        def drawPrimitives(pcollection):
            """
            Helper function.
            """
            self.figure.clf() # clear the current figure
            plot = self.figure.add_subplot(aspect='equal')

            # configure axes
            plot.set_xlim(-1/2, 1/2)
            plot.set_ylim(-1/2, 1/2)
            plot.axis('off')

            # place primitives from the collection on the plot
            for patch in pcollection:
                plot.add_patch(patch)




        self.vbuffer = [] # empy the vertex buffer every time we want to render the object

        # model matrix to move the object from its local space to the world space,
        # i.e. now all object's vertices will be defined relative to the center of the world
        model = object.setModelMatrix()

        # view matrix to go from the world space to the camera space,
        # i.e. now the object's coordinates will be defined relative to the camera
        view = camera.setLookAtMatrix()

        # projection matrix to map the coordinates to the screen space
        projection = camera.setProjectionMatrix()
        # joint matrix
        mvp = projection @ view @ model

        vshader = VertexShader()
        self.vbuffer += vshader.transform(mvp, object) # 2D coordinates on flat screen

        gshader = GeometryShader(geometry=Polygon)
        # generate collection of primitives to render the object
        collection = gshader.generate(self.vbuffer, vshader.zbuffer, object.attributes)

        # draw generated primitives to make up the object
        drawPrimitives(collection)


    def save_figure(self, fname="figure"):
        """
        Save the rendered figure in the file.

        Parameters
        ----------

        fname : str, default="figure"
            File name with the relative path to location the figure needs to be saved
        """
        save_path = Path(fname).absolute()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(save_path)

        


