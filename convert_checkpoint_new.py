import torch
import copy
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import iterative_closest_point as icp
import os
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from rbf_factory import RBFFactory
import tqdm


class Deformation(ABC):
    """
    Abstract class for generic deformation.
    This class should be inherited for the development of new deformation
    techniques.
    """
 
    @abstractmethod
    def __init__(self, value):
        pass

    @abstractmethod
    def __call__(self, src):
        pass


class RBF(Deformation):
    """
    Class that handles the Radial Basis Functions interpolation on the mesh
    points.

    :param numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :param numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
    :param func: the basis function to use in the transformation. Several basis
        function are already implemented and they are available through the
        :py:class:`~pygem.rbf.RBF` by passing the name of the right
        function (see class documentation for the updated list of basis
        function).  A callable object can be passed as basis function.
    :param float radius: the scaling parameter r that affects the shape of the
        basis functions.  For details see the class
        :class:`RBF`. The default value is 0.5.
    :param dict extra_parameter: the additional parameters that may be passed to
    	the kernel function. Default is None.
        
    :cvar numpy.ndarray weights: the matrix formed by the weights corresponding
        to the a-priori selected N control points, associated to the basis
        functions and c and Q terms that describe the polynomial of order one
        p(x) = c + Qx.  The shape is (*n_control_points+1+3*, *3*). It is
        computed internally.
    :cvar numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation.
    :cvar numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation.
    :cvar callable basis: the basis functions to use in the
        transformation.
    :cvar float radius: the scaling parameter that affects the shape of the
        basis functions.
    :cvar dict extra: the additional parameters that may be passed to the
        kernel function.
        
    :Example:

        >>> from pygem import RBF
        >>> import numpy as np
        >>> rbf = RBF(func='gaussian_spline')
        >>> xv = np.linspace(0, 1, 20)
        >>> yv = np.linspace(0, 1, 20)
        >>> zv = np.linspace(0, 1, 20)
        >>> z, y, x = np.meshgrid(zv, yv, xv)
        >>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
        >>> deformed_mesh = rbf(mesh)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 func='gaussian_spline',
                 radius=0.5,
                 extra_parameter=None):

        self.basis = func
        self.radius = radius

        if original_control_points is None:
            self.original_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.original_control_points = original_control_points

        if deformed_control_points is None:
            self.deformed_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.deformed_control_points = deformed_control_points

        self.extra = extra_parameter if extra_parameter else dict()

        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)


    @property
    def n_control_points(self):
        """
        Total number of control points.

        :rtype: int
        """
        return self.original_control_points.shape[0]

    @property
    def basis(self):
        """
        The kernel to use in the deformation.

        :getter: Returns the callable kernel
        :setter: Sets the kernel. It is possible to pass the name of the
            function (check the list of all implemented functions in the
            `pygem.rbf_factory.RBFFactory` class) or directly the callable
            function.
        :type: callable
        """
        return self.__basis

    @basis.setter
    def basis(self, func):
        if callable(func):
            self.__basis = func
        elif isinstance(func, str):
            self.__basis = RBFFactory(func)
        else:
            raise TypeError('`func` is not valid.')

    # def _get_weights(self, X, Y):
    #     """
    #     This private method, given the original control points and the deformed
    #     ones, returns the matrix with the weights and the polynomial terms, that
    #     is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
    #     (*n_control_points+1+3*, *3*).

    #     :param numpy.ndarray X: it is an n_control_points-by-3 array with the
    #         coordinates of the original interpolation control points before the
    #         deformation.
    #     :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
    #         coordinates of the interpolation control points after the
    #         deformation.

    #     :return: weights: the 2D array with the weights and the polynomial terms.
    #     :rtype: numpy.ndarray
    #     """
    #     npts, dim = X.shape
    #     H = np.zeros((npts + 3 + 1, npts + 3 + 1))
    #     H[:npts, :npts] = self.basis(cdist(X, X), self.radius, **self.extra)
    #     H[npts, :npts] = 1.0
    #     H[:npts, npts] = 1.0
    #     H[:npts, -3:] = X
    #     H[-3:, :npts] = X.T

    #     rhs = np.zeros((npts + 3 + 1, dim))
    #     rhs[:npts, :] = Y
    #     print(H.shape, rhs.shape, np.linalg.det(H)) 
    #     weights = np.linalg.solve(H, rhs)
    #     return weights
    
    def _get_weights(self, X, Y):
        """
        This private method, given the original control points and the deformed
        ones, returns the matrix with the weights and the polynomial terms, that
        is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
        (*n_control_points+1+3*, *3*).

        :param numpy.ndarray X: it is an n_control_points-by-3 array with the
            coordinates of the original interpolation control points before the
            deformation.
        :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
            coordinates of the interpolation control points after the
            deformation.

        :return: weights: the 2D array with the weights and the polynomial terms.
        :rtype: numpy.ndarray
        """
        npts, dim = X.shape
        H = np.zeros((npts, npts))
        H[:npts, :npts] = self.basis(cdist(X, X), self.radius, **self.extra)

        rhs = np.zeros((npts, dim))
        rhs[:npts, :] = Y
        # print(H.shape, rhs.shape, np.linalg.det(H)) 
        weights = np.linalg.solve(H, rhs)
        return weights

    def plot_points(self, filename=None):
        """
        Method to plot the control points. It is possible to save the resulting
        figure.

        :param str filename: if None the figure is shown, otherwise it is saved
            on the specified `filename`. Default is None.
        """
        fig = plt.figure(1)
        axes = fig.add_subplot(111, projection='3d')
        orig = axes.scatter(self.original_control_points[:, 0],
                            self.original_control_points[:, 1],
                            self.original_control_points[:, 2],
                            c='blue',
                            marker='o')
        defor = axes.scatter(self.deformed_control_points[:, 0],
                             self.deformed_control_points[:, 1],
                             self.deformed_control_points[:, 2],
                             c='red',
                             marker='x')

        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')

        plt.legend((orig, defor), ('Original', 'Deformed'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=10)

        # Show the plot to the screen
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)

    def compute_weights(self):
        """
        This method compute the weights according to the
        `original_control_points` and `deformed_control_points` arrays.
        """
        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)

    # def __call__(self, src_pts):
    #     """
    #     This method performs the deformation of the mesh points. After the
    #     execution it sets `self.modified_mesh_points`.
    #     """
    #     self.compute_weights()

    #     H = np.zeros((src_pts.shape[0], self.n_control_points + 3 + 1))
    #     H[:, :self.n_control_points] = self.basis(
    #         cdist(src_pts, self.original_control_points), 
    #         self.radius,
    #         **self.extra)
    #     H[:, self.n_control_points] = 1.0
    #     H[:, -3:] = src_pts
    #     return np.asarray(np.dot(H, self.weights))
    
    def __call__(self, src_pts):
        """
        This method performs the deformation of the mesh points. After the
        execution it sets `self.modified_mesh_points`.
        """
        self.compute_weights()

        H = np.zeros((src_pts.shape[0], self.n_control_points))
        H[:, :self.n_control_points] = self.basis(
            cdist(src_pts, self.original_control_points), 
            self.radius,
            **self.extra)
        # H[:, self.n_control_points] = 1.0
        # H[:, -3:] = src_pts
        return np.asarray(np.dot(H, self.weights))


def LDA(all_pcs, smooth_knn=10):
    base_pc = all_pcs[0]
    num_pts = base_pc.shape[0]
    device = base_pc.device
    num_steps = all_pcs.shape[0]
    nn_indices = torch.empty(num_pts, smooth_knn, dtype=torch.int64, device=device)
    for pt_idx in range(num_pts):
        # find the distance from the point at index i to all others points
        displacement_to_all_pts = (
            base_pc[pt_idx : pt_idx + 1, :].expand(num_pts, 3)
            - base_pc
        )
        _, inds = torch.topk(
            displacement_to_all_pts.pow(2).sum(dim=1),
            smooth_knn + 1,
            largest=False,
            sorted=True,
        )
        nn_indices[pt_idx, :] = inds[1:].type(torch.int64)

    for time_step in range(1, num_steps):
        avg_displacement = (
            all_pcs[time_step]
            .view(num_pts, 1, 3)
            .expand(num_pts, num_pts, 3)
            .gather(
                0,
                nn_indices
                .to(device)
                .unsqueeze(-1)
                .expand(num_pts, smooth_knn, 3),
            )
            - base_pc.view(num_pts, 1, 3)
            .expand(num_pts, num_pts, 3)
            .gather(
                0,
                nn_indices
                .to(device)
                .unsqueeze(-1)
                .expand(num_pts, smooth_knn, 3),
            )
        ).mean(dim=1)
        all_pcs[time_step] = avg_displacement + base_pc


nn_body = 24
kpnn_body = 24
body_kp = 256
wing_kp = 96
# wing_kp = 192
# body_kp = 512
# reg_D = True
reg_D = False
inp_case = 0
# force_smooth = True
force_smooth = False
force_smooth_full = True
smooth_knn = 50
# shift_kp = True
shift_kp = False
icp_instance = False
learnable_icp = True

# reg_boundary = True
reg_boundary = False
boundary_threshold = 0.05

rbf_radius = 10
per_part_reconstruction = True
# per_part_reconstruction = False

if body_kp == 256:
    target_exp_index = 9
elif body_kp == 512:
    target_exp_index = 11
elif body_kp == 192:
    target_exp_index = 10
elif body_kp == 96:
    target_exp_index = 8
else:
    raise ValueError("Invalid body_kp value")

if wing_kp == 96:
    target_exp_index = 12
# inp_case = 3
# base_path = "/NAS/spa176/pim/experiments/stand-stage-2/"
base_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-6-kp-nnwing5-nnbody250-wkpnn50-bkpnn256-regD/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-5-kp-nnwing5-nnbody190-wkpnn50-bkpnn192-regD-MSE/"
# save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn8-regD/"
# save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}/"
if reg_D:
    save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}-regD/"
else:
    if inp_case > 0:
        save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}-inp{inp_case}/"
    else:
        save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}/"
if force_smooth:
    save_path = save_path[:-1] + "-smooth/"
elif force_smooth_full:
    save_path = save_path[:-1] + "-smooth-full/"
if reg_boundary:
    save_path = save_path[:-1] + "-RB/"
# save_path = save_path[:-1] + f"-rbf-{rbf_radius}/"
save_path = save_path[:-1] + "-rbf/"
if per_part_reconstruction:
    save_path = save_path[:-1] + "-part/"
if shift_kp:
    save_path = save_path[:-1] + "-shift/"
if icp_instance:
    save_path = save_path[:-1] + "-icp/"
if learnable_icp:
    save_path = save_path[:-1] + "-Licp/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-9-kp-nnwing5-nnbody250-wkpnn50-bkpnn256-regD/"
model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-9-kp-nnwing5-nnbody16-wkpnn50-bkpnn16/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody90-wkpnn50-bkpnn8-regD/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody24-wkpnn50-bkpnn8-regD/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-4-kp-nnwing5-nnbody15-wkpnn50-bkpnn96-w1000/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-5-kp-nnwing5-nnbody15-wkpnn50-bkpnn192-w1000/model.pth"
# exp_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat_fps/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_96_cdw1000.0_rigidw1.0_nnwing5_nnbody90_kp_only_regD/"
# exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body96/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_8_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/"
# exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body96/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
if reg_D:
    exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/"
else:
    if inp_case > 0:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm_inp{inp_case}/"
    else:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
# if force_smooth:
#     exp_path = exp_path[:-1] + "_smooth/"
if reg_boundary:
    exp_path = exp_path[:-1] + f"_RB/"
    # exp_path = exp_path[:-1] + f"_RB{boundary_threshold}/"
# exp_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/L1_fix_wing_kp_192_body_kp_256_kpnn_50_bodykpnn_256_cdw1000.0_rigidw1.0_nnwing5_nnbody250_kp_only_regD/"
if wing_kp == 96:
    if icp_instance:
        print("Using ICP instance!!!")
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wingL{wing_kp}_wingR{wing_kp}_body{body_kp}_icp/L1_fix_wing_kp_{wing_kp}_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
    else:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wingL{wing_kp}_wingR{wing_kp}_body{body_kp}/L1_fix_wing_kp_{wing_kp}_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"

if learnable_icp:
    exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL{wing_kp}_wingR{wing_kp}_body{body_kp}/L1_fix_wing_kp_{wing_kp}_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"

# create folder if not exists
os.makedirs(save_path, exist_ok=True)
# copy the model fine to the save_path
# shutil.copy(model_path, save_path + "model.pth")

# state_dict = torch.load(base_path + "model_old_before_rotation.pth")
state_dict = torch.load(base_path + "model_no_rot_fix_name.pth")
# duplicate the model.pth to the same directory as model_old.pth using shutil
# shutil.copy(base_path + "model.pth", base_path + "model_no_rot_fix_name.pth")
step = list(state_dict.keys())[0]
state_dict = state_dict[step]

# print("Original state_dict keys:")
# print(state_dict.keys())

# print(a)

total_part_num = 3


bird_wing_indices = np.load("hummingbird_wing_indices.npy")
bird_body_indices = np.setdiff1d(
    np.arange(len(state_dict["points"])), bird_wing_indices
)

bird_wing_indices_right = np.load("hummingbird_wing_indices_right.npy")
bird_wing_indices_left = np.setdiff1d(
    np.arange(len(bird_wing_indices)), bird_wing_indices_right
)

if total_part_num == 3:
    tgt_part_indices = [
        bird_wing_indices[bird_wing_indices_left],
        bird_wing_indices[bird_wing_indices_right],
        bird_body_indices,
    ]
else:
    tgt_part_indices = [
        bird_wing_indices,
        bird_body_indices,
    ]

# total_part_num = 2
kps = []
tgt_pc = []
for part_idx in range(total_part_num):
    if shift_kp:
        part_pc = torch.load(
            os.path.join(exp_path, f"total_deformed_kps_p{part_idx}_shifted.pth")
        )
    else:
        part_pc = torch.load(os.path.join(exp_path, f"total_deformed_kps_p{part_idx}.pth"))
    LDA(part_pc, smooth_knn=20)
    kps.append(part_pc)

    if total_part_num == 3:
        if shift_kp:
            tgt_pc.append(
                np.load(
                    os.path.join(
                        os.path.dirname(os.path.dirname(exp_path)),
                        f"part{part_idx}_tgt_pc_ori.npy",
                    )
                )
            )
        else:
            tgt_pc.append(np.load(os.path.join(os.path.dirname(os.path.dirname(exp_path)),f"part{part_idx}_tgt_pc.npy")))
if not per_part_reconstruction:
    kps = torch.cat(kps, dim=1)

rbf_kernel = "polyharmonic_spline"
ext_params = {"k": 3} if rbf_kernel == "polyharmonic_spline" else {}
scale = 10.0

if total_part_num == 2:
    tgt_pc = np.load("/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body256/rotated_bird.npy") * scale
else:
    if not per_part_reconstruction:
        tgt_pc = np.concatenate(tgt_pc, axis=0) * scale

if per_part_reconstruction:
    total_deformed_pcs = [
        torch.from_numpy(np.concatenate(tgt_pc, axis=0) * scale)
        .float()
        .to(state_dict["points"].device)
    ]
    for cur_step in tqdm.tqdm(range(1, len(kps[0]))):
        cur_deformed_pcs = []
        for part_idx in range(total_part_num):
            rbf = RBF(
                original_control_points=kps[part_idx][0].cpu().numpy() * scale,
                deformed_control_points=kps[part_idx][cur_step].cpu().numpy() * scale,
                radius=rbf_radius,
                func=rbf_kernel,
                extra_parameter=ext_params,
            )
            # # Deform the surface points
            deformed_full_point_cloud = rbf(tgt_pc[part_idx] * scale)
            cur_deformed_pcs.append(
                torch.from_numpy(deformed_full_point_cloud)
                .float()
                .to(state_dict["points"].device)
            )
        total_deformed_pcs.append(torch.cat(cur_deformed_pcs, dim=0))

else:
    total_deformed_pcs = [torch.from_numpy(tgt_pc).float().to(state_dict["points"].device)]

    # rbf_kernel = 'gaussian_spline'
    for cur_step in tqdm.tqdm(range(1, len(kps))):
        ext_params = {'k': 3} if rbf_kernel == 'polyharmonic_spline' else {}
        rbf = RBF(original_control_points=kps[0].cpu().numpy() * scale, deformed_control_points=kps[cur_step].cpu().numpy() * scale, radius=rbf_radius, func=rbf_kernel, extra_parameter=ext_params)
        # # Deform the surface points
        deformed_full_point_cloud = rbf(tgt_pc)
        total_deformed_pcs.append(torch.from_numpy(deformed_full_point_cloud).float().to(state_dict["points"].device))
total_deformed_pcs = torch.stack(total_deformed_pcs, dim=0)


if total_part_num == 3:
    # adjust the ordering of pc_feats, points_influ_scores
    new_pc_feats = [state_dict["pc_feats"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
    new_pc_feats = torch.cat(new_pc_feats, dim=0)
    state_dict["pc_feats"] = new_pc_feats

    new_points_influ_scores = [state_dict["points_influ_scores"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
    new_points_influ_scores = torch.cat(new_points_influ_scores, dim=0)
    state_dict["points_influ_scores"] = new_points_influ_scores

# align the deformed_pc with original pc
converged, rmse, Xt, RTs, t_history = icp(
    total_deformed_pcs[0].unsqueeze(0), state_dict["points"].unsqueeze(0), max_iterations=300
)
print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# deformed_pc = Xt.squeeze(0)

R = RTs.R
T = RTs.T

total_deformed_pcs = torch.bmm(total_deformed_pcs, R.expand(total_deformed_pcs.shape[0], 3, 3)) + T[:, None, :].expand(total_deformed_pcs.shape[0], -1, 3)

save_pc_name = f"total_deformed_pc_rbf.pth"
# save the transformed deformed_pc
torch.save(total_deformed_pcs, save_path + save_pc_name)

state_dict["points"] = total_deformed_pcs[0]
save_sd = {step: state_dict}
torch.save(save_sd, save_path + "model.pth")

# # plot the deformed_pc in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_pc = deformed_pc.cpu().numpy()
# ax.scatter(plot_pc[:, 0], plot_pc[:, 1], plot_pc[:, 2])
# # limit the axis to the same range as the original pc
# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-5, 8)
# plt.savefig("deformed_pc_4.png")
# plt.close()
# # plot the original pc in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_pc = state_dict["points"].cpu().numpy()
# ax.scatter(plot_pc[:, 0], plot_pc[:, 1], plot_pc[:, 2])
# # limit the axis to the same range as the original pc
# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-5, 8)
# plt.savefig("original_pc.png")
# plt.close()

# exit(0)


# ################ For saving a new checkpoint with the deformed_pc ################
# # for each point in the deformed_pc, find the closest point in the original pc and copy the influence score and pc_feat from the original pc to the deformed pc

# num_pts = deformed_pc.shape[0]
# new_points_influ_scores = torch.empty(num_pts, 1, device=state_dict["points_influ_scores"].device)
# new_pc_feat = torch.empty(
#     num_pts,
#     state_dict["pc_feats"].shape[1],
#     device=state_dict["pc_feats"].device,
# )

# print("deformed_pc shape", deformed_pc.shape)
# print("state_dict points shape", state_dict["points"].shape)
# print("points_influ_scores shape", state_dict["points_influ_scores"].shape)
# print("pc_feats shape", state_dict["pc_feats"].shape)

# for i in range(deformed_pc.shape[0]):
#     deformed_point = deformed_pc[i]
#     diff = state_dict["points"] - deformed_point
#     dist = torch.norm(diff, dim=1)
#     min_dist, min_idx = torch.min(dist, dim=0)
#     new_points_influ_scores[i] = state_dict["points_influ_scores"][min_idx]
#     new_pc_feat[i] = state_dict["pc_feats"][min_idx]


# state_dict["points_influ_scores"] = new_points_influ_scores
# state_dict["pc_feats"] = new_pc_feat
# state_dict["points"] = deformed_pc.to(state_dict["points"].device)

# save_sd = {step: state_dict}
# torch.save(save_sd, save_path + "model.pth")
# #################################################################################
