import os, sys
sys.path.append(os.path.join(os.getcwd()))
import pyvista
import time
import torch
import numpy as np
from lib.utils.visualizer3d import Visualizer3D
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from motion_infiller.data.amass_dataset import AMASSDataset
from torch.utils.data import DataLoader
# from pyvista.plotting.tools import parse_color
from vtk import vtkTransform
from lib.utils.torch_transform import quat_apply, quat_between_two_vec, quaternion_to_angle_axis, angle_axis_to_quaternion


class SMPLActor():  # Responsável por criar um ator para visualização da malha SMPL de uma pessoa
    '''
        plotter: O objeto plotter do PyVista usado para renderizar a malha.
        vertices: Os vértices da malha SMPL da pessoa. Esses vértices devem ser um array ou objeto do PyVista que represente as coordenadas 3D dos vértices.
        faces: As faces da malha SMPL da pessoa. Essas faces devem ser um array ou objeto do PyVista que represente a conectividade entre os vértices da malha.
        visible (opcional): Um valor booleano que indica se o ator deve ser visível ou não. O padrão é True.
        color (opcional): A cor do ator. Pode ser uma string representando um nome de cor ou um valor RGB. O padrão é 'white'.
    '''
    def __init__(self, pl, verts, faces, color='#FF8A82', visible=True):
        self.pl = pl
        self.verts = verts
        self.face = faces   # faces contem o index do vértice utilizado em verts (utilizada quadrados em vez de triangulos)
        self.mesh = pyvista.PolyData(verts, faces)
        self.actor = self.pl.add_mesh(self.mesh, color=color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
        # self.actor = self.pl.add_mesh(self.mesh, color=color, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        self.set_visibility(visible)

    def update_verts(self, new_verts):
        self.mesh.points[...] = new_verts
        self.mesh.compute_normals(inplace=True)

    def set_opacity(self, opacity):
        self.actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        self.actor.SetVisibility(flag)

    # def set_color(self, color):
    #     rgb_color = parse_color(color)
    #     self.actor.GetProperty().SetColor(rgb_color)


class SkeletonActor():

    def __init__(self, pl, joint_parents, joint_color='green', bone_color='yellow', joint_radius=0.03, bone_radius=0.02, visible=True):
        self.pl = pl
        self.joint_parents = joint_parents
        self.joint_meshes = []
        self.joint_actors = []
        self.bone_meshes = []
        self.bone_actors = []
        self.bone_pairs = []
        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = pyvista.Sphere(radius=joint_radius, center=(0, 0, 0), theta_resolution=10, phi_resolution=10)
            # joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
            joint_actor = self.pl.add_mesh(joint_mesh, color=joint_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
            self.joint_meshes.append(joint_mesh)
            self.joint_actors.append(joint_actor)
            # bone
            if pa >= 0:
                bone_mesh = pyvista.Cylinder(radius=bone_radius, center=(0, 0, 0), direction=(0, 0, 1), resolution=30)
                # bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, pbr=True, metallic=0.0, roughness=0.3, diffuse=1)
                bone_actor = self.pl.add_mesh(bone_mesh, color=bone_color, ambient=0.3, diffuse=0.5, specular=0.8, specular_power=5, smooth_shading=True)
                self.bone_meshes.append(bone_mesh)
                self.bone_actors.append(bone_actor)
                self.bone_pairs.append((j, pa))
        self.set_visibility(visible)

    def update_joints(self, jpos):
        # joint
        for actor, pos in zip(self.joint_actors, jpos):
            trans = vtkTransform()
            trans.Translate(*pos)
            actor.SetUserTransform(trans)
        # bone
        vec = []
        for actor, (j, pa) in zip(self.bone_actors, self.bone_pairs):
            vec.append((jpos[j] - jpos[pa]))
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quaternion_to_angle_axis(quat_between_two_vec(torch.tensor([0., 0., 1.]).expand_as(vec), vec)).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)
        
        for actor, (j, pa), angle_i, axis_i, dist_i in zip(self.bone_actors, self.bone_pairs, angle, axis, dist):
            trans = vtkTransform()
            trans.Translate(*(jpos[pa] + jpos[j]) * 0.5)
            trans.RotateWXYZ(np.rad2deg(angle_i), *axis_i)
            trans.Scale(1, 1, dist_i)
            actor.SetUserTransform(trans)

    def set_opacity(self, opacity):
        for actor in self.joint_actors:
            actor.GetProperty().SetOpacity(opacity)
        for actor in self.bone_actors:
            actor.GetProperty().SetOpacity(opacity)

    def set_visibility(self, flag):
        for actor in self.joint_actors:
            actor.SetVisibility(flag)
        for actor in self.bone_actors:
            actor.SetVisibility(flag)

    # def set_color(self, color):
    #     rgb_color = parse_color(color)
    #     for actor in self.joint_actors:
    #         actor.GetProperty().SetColor(rgb_color)
    #     for actor in self.jbone_actors:
    #         actor.GetProperty().SetColor(rgb_color)



class SMPLVisualizer(Visualizer3D):

    def __init__(self, generator_func=None, device=torch.device('cpu'), show_smpl=True, show_skeleton=False, sample_visible_alltime=False, **kwargs):
        super().__init__(**kwargs)
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)
        faces = self.smpl.faces.copy()       
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = self.smpl.parents.cpu().numpy()
        self.generator_func = generator_func
        self.smpl_motion_generator = None
        self.device = device
        self.sample_visible_alltime = sample_visible_alltime
        
    def update_smpl_seq(self, smpl_seq=None, mode='gt'):
        if smpl_seq is None:
            try:
                smpl_seq = next(self.smpl_motion_generator)
            except:
                self.smpl_motion_generator = self.generator_func()
                smpl_seq = next(self.smpl_motion_generator)
        self.smpl_seq = smpl_seq

        self.smpl_verts = None

        if mode == 'gt':
            key = ''
        elif mode == 'sample':
            key = 'infer_out_'
        else:
            key = 'recon_out_'

        if f'{key}pose' in smpl_seq:
            assert smpl_seq['shape'].shape[0] == 1

            print(f'use {mode}')
            pose = smpl_seq[f'{key}pose']
            if mode == 'sample':
                pose = pose.squeeze(0)

            if f'{key}trans' in smpl_seq:
                trans = smpl_seq[f'{key}trans']
            else:
                trans = smpl_seq['trans'].repeat((pose.shape[0], 1, 1))
            if mode == 'sample':
                trans = trans.squeeze(0)

            shape = smpl_seq['shape'].repeat((pose.shape[0], 1, 1))
            
            # print(pose[..., :3].view(-1, 3))
            orig_pose_shape = pose.shape
            self.smpl_motion = self.smpl(
                global_orient=pose[..., :3].view(-1, 3),
                body_pose=pose[..., 3:].view(-1, 69),
                betas=torch.zeros_like(shape.view(-1, 10)),     # TODO
                root_trans = trans.view(-1, 3),
                return_full_pose=True,
                orig_joints=True
            )

            self.smpl_verts = self.smpl_motion.vertices.reshape(*orig_pose_shape[:-1], -1, 3)
            self.smpl_joints = self.smpl_motion.joints.reshape(*orig_pose_shape[:-1], -1, 3)
        
        if f'{key}joint_pos' in smpl_seq:
            print(f'use {mode} joint pos')
            joints = smpl_seq[f'{key}joint_pos']
            if mode == 'sample':
                joints = joints.squeeze(0)

            if f'{key}trans' in smpl_seq:
                trans = smpl_seq[f'{key}trans']
            else:
                trans = smpl_seq['trans'].repeat((joints.shape[0], 1, 1))
            
            if f'{key}orient' in smpl_seq:
                orient = smpl_seq[f'{key}orient']
            else:
                orient = smpl_seq['pose'][..., :3].repeat((joints.shape[0], 1, 1))

            if mode == 'sample':
                trans = trans.squeeze(0)
                orient = orient.squeeze(0)

            joints = torch.cat([torch.zeros_like(joints[..., :3]), joints], dim=-1).view(*joints.shape[:-1], -1, 3)
            orient_q = angle_axis_to_quaternion(orient).unsqueeze(-2).expand(joints.shape[:-1] + (4,))
            joints_world = quat_apply(orient_q, joints) + trans.unsqueeze(-2)
            self.smpl_joints = joints_world

        self.fr = 0
        self.num_fr = self.smpl_joints.shape[1]
        self.mode = mode
        self.vis_mask = self.smpl_seq['frame_mask'][0].cpu().numpy()

    def init_camera(self):
        super().init_camera()

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        self.floor_mesh.points[:, 2] -= 0.08
        self.update_smpl_seq(init_args.get('smpl_seq', None), init_args.get('mode', 'gt'))
        self.num_actors = init_args.get('num_actors', self.smpl_joints.shape[0])
        if self.show_smpl and self.smpl_verts is not None:
            vertices = self.smpl_verts[0, 0].cpu().numpy()
            self.smpl_actors = [SMPLActor(self.pl, vertices, self.smpl_faces) for _ in range(self.num_actors)]
        if self.show_skeleton:
            self.skeleton_actors = [SkeletonActor(self.pl, self.smpl_joint_parents) for _ in range(self.num_actors)]
        
    def update_camera(self, interactive):
        root_pos = self.smpl_joints[0, self.fr, 0].cpu().numpy()
        roll = self.pl.camera.roll
        view_vec = np.asarray(self.pl.camera.position) - np.asarray(self.pl.camera.focal_point)
        new_focal = np.array([root_pos[0], root_pos[1], 0.8])
        new_pos = new_focal + view_vec
        self.pl.camera.up = (0, 0, 1)
        self.pl.camera.focal_point = new_focal.tolist()
        self.pl.camera.position = new_pos.tolist()
        # self.pl.camera.roll = roll   # don't set roll

    def update_scene(self):
        super().update_scene()
        visible = self.vis_mask[self.fr] == 1.0
        all_visible = np.all(self.vis_mask == 1.0)

        if self.show_smpl and self.smpl_verts is not None:
            if all_visible:
                full_opacity = 0.5 
            elif self.show_skeleton or self.num_actors > 1:
                full_opacity = 0.7
            else:
                full_opacity = 1.0
            opacity = full_opacity if visible else 0.5
            
            for i, actor in enumerate(self.smpl_actors):
                if visible and i > 0 and not self.sample_visible_alltime:
                    actor.set_visibility(False)
                else:
                    actor.set_visibility(True)
                    actor.update_verts(self.smpl_verts[i, self.fr].cpu().numpy())
                actor.set_opacity(opacity)

        if self.show_skeleton:
            if all_visible:
                full_opacity = 0.5 
            elif self.show_skeleton or self.num_actors > 1:
                full_opacity = 0.7
            else:
                full_opacity = 1.0
            opacity = full_opacity if visible else 0.4

            for i, actor in enumerate(self.skeleton_actors):
                if visible and i > 0 and not self.sample_visible_alltime:
                    actor.set_visibility(False)
                else:
                    actor.set_visibility(True)
                    actor.update_joints(self.smpl_joints[i, self.fr].cpu().numpy())
                actor.set_opacity(opacity)

    def setup_key_callback(self):
        super().setup_key_callback()

        def next_data():
            self.update_smpl_seq()

        def reset_camera():
            self.init_camera()

        self.pl.add_key_event('z', next_data)
        self.pl.add_key_event('t', reset_camera)
