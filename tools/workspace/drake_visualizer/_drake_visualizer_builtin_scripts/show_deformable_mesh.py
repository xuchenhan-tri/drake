# Note that this script runs in the main context of drake-visualizer,
# where many modules and variables already exist in the global scope.

from director import lcmUtils
from director import applogic
from director import objectmodel as om
from director import visualization as vis
import director.vtkAll as vtk

import drake as lcmdrakemsg

from _drake_visualizer_builtin_scripts import scoped_singleton_func


class DeformableMeshVisualizer:
    def __init__(self):
        self._folder_name = 'DeformableMesh'
        self._name = "Deformable Mesh Visualizer"
        self._enabled = False
        # The subscribers: one for init and one for update.
        self._subs = None
        self._mesh_name = None
        self._vertex_count = -1
        self._polyData = None
        self._poly_item = None

        self.set_enabled(True)

    def add_subscriber(self):
        if self._subs is not None:
            return

        self._subs = []
        self._subs.append(lcmUtils.addSubscriber(
            'DEFORMABLE_MESH_INIT',
            messageClass=lcmdrakemsg.lcmt_deformable_tri_mesh_init,
            callback=self.handle_init_message))

        self._subs.append(lcmUtils.addSubscriber(
            'DEFORMABLE_MESH_UPDATE',
            messageClass=lcmdrakemsg.lcmt_deformable_tri_mesh_update,
            callback=self.handle_update_message))

        print(self._name + " subscribers added.")

    def remove_subscriber(self):
        if self._subs is None:
            return

        for sub in self._subs:
            lcmUtils.removeSubscriber(sub)
        self._subs = None
        print(self._name + " subscribers removed.")

    def is_enabled(self):
        return self._enabled

    def set_enabled(self, enable, clear=True):
        print("DeformableVisualizer set_enabled", enable)
        self._enabled = enable
        if enable:
            self.add_subscriber()
        else:
            self.remove_subscriber()
            # Removes the folder completely and resets the known meshes.
            om.removeFromObjectModel(om.findObjectByName(self._folder_name))
            self._poly_item = None

    def handle_init_message(self, msg):
        """Creates the polydata for the deformable mesh specified in msg."""
        folder = om.getOrCreateContainer(self._folder_name)
        if self._poly_item:
            om.removeFromObjectModel(self._poly_item)

        self._mesh_name = msg.mesh_name
        # Initial vertex positions are garbage; meaningful values will be set
        # by the update message.
        points = vtk.vtkPoints()
        self._vertex_count = msg.num_vertices
        for i in range(self._vertex_count):
            points.InsertNextPoint(0, 0, 0)
        triangles = vtk.vtkCellArray()
        for tri in msg.tris:
            triangles.InsertNextCell(3)
            for i in tri.vertices:
                triangles.InsertCellPoint(i)
        self._polyData = vtk.vtkPolyData()
        self._polyData.SetPoints(points)
        self._polyData.SetPolys(triangles)
        self._poly_item = vis.showPolyData(self._polyData, self._mesh_name,
                                           parent=folder)
        self._poly_item.setProperty('Surface Mode', 'Surface with edges')

    def handle_update_message(self, msg):
        """Updates vertex data for the deformable mesh specified in msg."""
        if self._polyData is None:
            print("Received a deformable mesh update message for mesh '{}', "
                  "but no such mesh has been initialized."
                    .format(msg.mesh_name))
            return
        if msg.mesh_name != self._mesh_name:
            print("The deformable mesh update message contains data for a mesh "
                  "named '{}', expected name '{}'."
                  .format(msg.mesh_name, self._mesh_name))
            return
        if msg.data_size != self._vertex_count * 3:
            print("The deformable mesh update message contains data for {} "
                  "vertices; exppected {}."
                  .format(msg.data_size // 3, self._vertex_count))
            return
        points = vtk.vtkPoints()
        i = 0
        while i < msg.data_size:
            points.InsertNextPoint(msg.data[i], msg.data[i + 1],
                                   msg.data[i + 2])
            i = i + 3
        # TODO(SeanCurtis-TRI): Instead of creating a new set of points and
        #  stomping on the old; can I just update the values? That might improve
        #  performance.
        self._polyData.SetPoints(points)
        self._poly_item.setPolyData(self._polyData)
        self._poly_item._renderAllViews()


@scoped_singleton_func
def init_visualizer():
    # Create a visualizer instance.
    my_visualizer = DeformableMeshVisualizer()
    # Adds to the "Tools" menu.
    applogic.MenuActionToggleHelper(
        'Tools', my_visualizer._name,
        my_visualizer.is_enabled, my_visualizer.set_enabled)
    return my_visualizer


# Activate the plugin if this script is run directly; store the results to keep
# the plugin objects in scope.
if __name__ == "__main__":
    deform_viz = init_visualizer()
