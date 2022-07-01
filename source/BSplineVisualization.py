
import os
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl import operations

from geomdl import multi
from geomdl.visualization import VisMPL
import numpy as np
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.pyplot as plt
import helper


def generateSurface(points):
    surf = BSpline.Surface()

    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts2d = points.tolist()
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

    surf.delta = 0.025
    surf.evaluate()

    return surf


def visualizeSingleFrame(leftBM5, rightBM5, zSubdivisions, leftPoints = None, rightPoints = None):
        pointsLeft = leftBM5.reshape(zSubdivisions, -1 ,3)
        pointsLeft = np.concatenate([pointsLeft, pointsLeft[:, :1, :]], axis=1)

        pointsRight = rightBM5.reshape(zSubdivisions, -1 ,3)
        pointsRight = np.concatenate([pointsRight, pointsRight[:, :1, :]], axis=1)

        surfLeft = generateSurface(pointsLeft)
        surfRight = generateSurface(pointsRight)

        #uvMidPoint = helper.midPointMethod(surfLeft, leftPoints[i])
        #projectionPoints = surfLeft.evaluate_list(uvMidPoint.tolist())
        #test = np.concatenate([np.expand_dims(leftPoints[i], 1), np.expand_dims(projectionPoints, 1)], axis=1)

        #v_values = np.linspace(0.05, 0.3, 40)
        #u_values = np.ones((40, 1))*0.4
        #uv_values = np.concatenate([u_values, np.expand_dims(v_values, -1)], axis=1)

        #geodesicLeft =  surfLeft.evaluate_list(uv_values.tolist())
        #geodesicRight = surfRight.evaluate_list(uv_values.tolist())

        surfaces = multi.SurfaceContainer([surfLeft, surfRight])

        # Set number of samples for all split surfaces
        surfaces.sample_size = 30

        # Plot the control point grid and the evaluated surface
        vis_comp = VisMPL.VisSurface(display_axes=False)
        surfaces.vis = vis_comp
        component = surfaces.render(colormap=[cm.inferno, cm.inferno])
        #surfaces.vis.add(geodesicLeft, "geodesic", name="Geodesic Left", color="#333333")
        #surfaces.vis.add(geodesicRight, "geodesic", name="Geodesic Right", color="#333333")
        surfaces.vis.add(leftPoints, "extras", name="Points Left", color="#FF0000")
        surfaces.vis.add(rightPoints, "extras", name="Points Right", color="#00FF00")
        component.render(display_plot=True)
        pass


def visualizeBM5(leftBM5, rightBM5, zSubdivisions, leftPoints = None, rightPoints = None, filename = "", plot=True):

    for i in tqdm(range(leftBM5.shape[0])):
        pointsLeft = leftBM5[i].reshape(zSubdivisions, -1 ,3)
        pointsLeft = np.concatenate([pointsLeft, pointsLeft[:, :1, :]], axis=1)

        pointsRight = rightBM5[i].reshape(zSubdivisions, -1 ,3)
        pointsRight = np.concatenate([pointsRight, pointsRight[:, :1, :]], axis=1)

        surfLeft = generateSurface(pointsLeft)
        surfRight = generateSurface(pointsRight)

        #uvMidPoint = helper.midPointMethod(surfLeft, leftPoints[i])
        #projectionPoints = surfLeft.evaluate_list(uvMidPoint.tolist())
        #test = np.concatenate([np.expand_dims(leftPoints[i], 1), np.expand_dims(projectionPoints, 1)], axis=1)

        #v_values = np.linspace(0.05, 0.3, 40)
        #u_values = np.ones((40, 1))*0.4
        #uv_values = np.concatenate([u_values, np.expand_dims(v_values, -1)], axis=1)

        #geodesicLeft =  surfLeft.evaluate_list(uv_values.tolist())
        #geodesicRight = surfRight.evaluate_list(uv_values.tolist())

        surfaces = multi.SurfaceContainer([surfLeft, surfRight])

        # Set number of samples for all split surfaces
        surfaces.sample_size = 30

        # Plot the control point grid and the evaluated surface
        vis_comp = VisMPL.VisSurface(display_axes=False)
        surfaces.vis = vis_comp
        component = surfaces.render(colormap=[cm.inferno, cm.inferno])
        #surfaces.vis.add(geodesicLeft, "geodesic", name="Geodesic Left", color="#333333")
        #surfaces.vis.add(geodesicRight, "geodesic", name="Geodesic Right", color="#333333")
        surfaces.vis.add(leftPoints[i], "extras", name="Points Left", color="#FF0000")
        surfaces.vis.add(rightPoints[i], "extras", name="Points Right", color="#00FF00")
        try:
            os.mkdir(filename)
        except:
            pass

        component.render(display_plot=plot, fig_save_as=filename + "/{0:05d}.png".format(i))
        pass


def visualizeSingleBM5(bm5, points, zSubdivisions):
    geodesic_xy = list()
    for i in range(bm5.shape[0]):
        pointsLeft = bm5[i].reshape(zSubdivisions, -1 ,3)
        pointsLeft = np.concatenate([pointsLeft, pointsLeft[:, :1, :]], axis=1)

        surfLeft = generateSurface(pointsLeft)

        #v_values = np.linspace(0.00, 0.2, 40)
        #u_values = np.ones((40, 1))*0.4
        #uv_values = np.concatenate([u_values, np.expand_dims(v_values, -1)], axis=1)

        #geodesic =  surfLeft.evaluate_list(uv_values.tolist())
        #geodesic_xy.append(np.array(geodesic)[:, 0:2])
        surfaces = multi.SurfaceContainer([surfLeft])

        # Set number of samples for all split surfaces
        surfaces.sample_size = 20

        # Plot the control point grid and the evaluated surface
        if i == 0:
            vis_comp = VisMPL.VisSurface(display_axes=False)
            surfaces.vis = vis_comp
            component = surfaces.render(colormap=[cm.inferno])
            surfaces.vis.add(points[i], "extras", name="Points Left", color="#FF0000")
            component.render(display_plot=True)
    
    #fig, ax = plt.subplots()
    #for geodesic in geodesic_xy:
    #    ax.plot(geodesic[:, 0], geodesic[:, 1])
    #plt.show()