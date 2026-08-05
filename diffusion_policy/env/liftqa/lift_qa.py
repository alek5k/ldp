from enum import Enum
from pathlib import Path

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.controllers.parts.controller_factory import load_part_controller_config




MAX_GOAL_VISITS = 6

class PlacardMaterial(CustomMaterial):
    def __init__(self, texture_path: Path, unique_tex_name: str, unique_mat_name: str, tex_attrib=None, mat_attrib=None, shared=False):
        # Call CustomMaterial init with path instead of rgba
        super().__init__(
            texture=texture_path.as_posix(),  # acts like a path, not in TEXTURES. Robosuite looks for a '/' so we need to use posix
            tex_name=unique_tex_name,
            mat_name=unique_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=shared,
        )

class GoalPhase(Enum):
    START = 0
    AT_QA = 1
    FINISHED_WAIT_AT_QA = 2
    RETURNED_TO_START = 3
    FINISHED_WAIT_AT_START = 4


class LiftQA(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        initial_robot_joints=None,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        constrain_motion=True,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.initial_robot_joints = initial_robot_joints
        self.motion_constrained = constrain_motion
        self.control_freq = control_freq

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )
        self.set_xml_processor(self._modify_xml)

        

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _modify_xml(self, xml):
        import xml.etree.ElementTree as ET
        from copy import deepcopy
        root = ET.fromstring(xml)
        xml = ET.tostring(root, encoding="utf8").decode("utf8")
        return xml


    def _create_custom_objects(self):
        
        self.custom_objects = []
        self.custom_object_placements = {}

        def _make_object(name: str, size: list, rel_path: str, rgba: list, xyz: list = [0,0,0], rpy: list = [0,0,0], objtype = "box"):
            mtl = None
            if rel_path:
                mtl = PlacardMaterial(
                    texture_path=Path(__file__).parent.joinpath(rel_path),
                    unique_tex_name=f"{name}_tex",
                    unique_mat_name=f"{name}_mat",
                )
            if objtype == "box":
                placard = BoxObject(
                    name=f"{name}_placard",
                    size=size,
                    rgba=rgba,
                    material=mtl,
                    obj_type="visual",
                    joints=None
                )
            elif objtype == "cylinder":
                from robosuite.models.objects import CylinderObject
                placard = CylinderObject(
                    name=f"{name}_placard",
                    size=size[:2],
                    rgba=rgba,
                    material=mtl,
                    obj_type="visual",
                    joints=None
                )
            rpy = [np.deg2rad(angle) for angle in rpy]
            rotation = T.euler2mat(rpy)  # rotate to be vertical
            pose = T.make_pose(np.array(xyz), rotation)
            self.custom_objects.append(placard)
            self.custom_object_placements[placard.name] = pose
            return placard
        
        
            # x_range=[-0.03, -0.03], # more positive -> away from robot
            # y_range=[-0.3, -0.3],
        
        camera_placard = _make_object(
            name="camera", 
            size=[0.1, 0.1, 0.001], 
            # rel_path="camera_icon.png", 
            rel_path=None,
            rgba=[0.4, 0.7, 1.0, 1.0], 
            xyz=self.table_offset + np.array([-0.16, 0.2, 0.3]),
            rpy=[90, 90, 0])
        
        camera_placard_active = _make_object(
            name="camera_active", 
            size=[0.1, 0.1, 0.001], 
            # rel_path="camera_icon.png", 
            rel_path=None,
            rgba=[1, 1, 1, 1], # rgba=[0, 1, 0, 1], 
            xyz=self.table_offset + np.array([100, 100, 100]),  # start out of view
            rpy=[90, 90, 0])
        camera_placard_active.visible = False  # start inactive
        
        
        camera_body1 = _make_object(
            name="camera_fraeme", 
            size=[0.025, 0.06, 0.04], 
            rel_path=None, 
            rgba=[0.5, 0.5, 0.5, 1], 
            xyz=self.table_offset + np.array([-0.15, 0.2, 0.3]),
            rpy=[0, 0, 0])
        camera_body2 = _make_object(
            name="camera_fraeme2", 
            size=[0.03, 0.03], 
            rel_path=None, 
            rgba=[0.1, 0.1, 0.1, 1], 
            xyz=self.table_offset + np.array([-0.13, 0.2, 0.3]),
            rpy=[0, 90, 0],
            objtype="cylinder")
        camera_body3 = _make_object(
            name="camera_fraeme3", 
            size=[0.027, 0.027], 
            rel_path=None, 
            rgba=[0.678, 0.847, 0.902, 1], 
            xyz=self.table_offset + np.array([-0.125, 0.2, 0.3]),
            rpy=[0, 90, 0],
            objtype="cylinder")
        
        target_placard = _make_object(
            name="target", 
            size=[0.1, 0.1, 0.001], 
            rel_path=None, 
            rgba=[0, 1, 0, 1], 
            xyz=self.table_offset + self.cube_table_offset + np.array([0, 0, -0.01]), 
            rpy=[0, 0, 0])
        



    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.020, 0.020, 0.020],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            density=200, # extremely light. 1000 is water.
            # friction=0.6,
            material=redwood,
            rng=self.rng,
        )

        self.cube_table_offset = [-0.03, -0.3, 0.01]
        self.cube_y_variance = 0.07
        self._create_custom_objects()

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[self.cube_table_offset[0], self.cube_table_offset[0]], # more positive -> away from robot
                y_range=[self.cube_table_offset[1] - self.cube_y_variance, self.cube_table_offset[1] + self.cube_y_variance],
                rotation=[0, 0],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube] + self.custom_objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        for obj in self.custom_objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
    
    

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        
        for obj_id, pose in self.custom_object_placements.items():
            pos, quat = T.mat2pose(pose)
            self.sim.model.body_pos[self.obj_body_id[obj_id]] = pos
            self.sim.model.body_quat[self.obj_body_id[obj_id]] = quat


    def reset(self):
        self.goal_stage = 0
        self.wait_timer = 0.0  # Tracks time spent at goal
        self.wait_times_each_visit = []
        self._current_visit_timer = 0.0
        self._visit_active = False
        self.finished_timer = 0.0 # tracks time spent back at the start
        self.wait_duration = 2.0  # Seconds to wait at goal

        # Soft reset workaround for mjviewer
        if self.renderer == "mjviewer":
            self.renderer = "mjviewer2" # mujoco interface checks that renderer is "mjviewer" for hard reset
            obs = super().reset()
            self.renderer = "mjviewer"
        else:
            obs = super().reset()

        # move the agentview camera to frontview + some offset
        frontview_id = self.sim.model.camera_name2id("frontview")
        frontview_pos = np.copy(self.sim.model.cam_pos[frontview_id])  # 1.6, 0, 1.45
        frontview_quat = np.copy(self.sim.model.cam_quat[frontview_id])  # [0.56084189, 0.43064645, 0.43064645, 0.56084189]
        frontview_pos[0] -= 0.85  # move closer along x axis
        frontview_pos[1] -= 0.075  # move closer along y axis
        frontview_pos[2] -= 0.2  # move closer along z axis
        agentview_id = self.sim.model.camera_name2id("agentview")
        self.sim.model.cam_pos[agentview_id] = np.array(frontview_pos)
        self.sim.model.cam_quat[agentview_id] = np.array(frontview_quat)

        # have to call step before/after otherwise changes don't take effect
        self.initial_robot_x = 0.0 # just define first to avoid error
        self.last_x = 0.0
        self.last_z = 0.0
        self.step(np.zeros(self.action_dim))  # step once to initialize
        if self.initial_robot_joints is not None:
            self.robots[0].set_robot_joint_positions(self.initial_robot_joints)
        obs, _, _, _ = self.step(np.zeros(self.action_dim)) 
        self.initial_robot_x = obs['robot0_eef_pos'][0]

        # Step a few more times to let the sim settle
        for _ in range(5):
            obs, _, _, _ = self.step(np.zeros(self.action_dim))

        self.sim.data.time = 0.0
        self.cur_time = 0.0
        self.timestep = 0
        
        obs_extra = self._get_obs_custom(obs)
        obs.update(obs_extra)


        return obs

    def _get_obs_custom(self, obs):
        img_flipped = obs['agentview_image'][::-1, :, :] # flip vertical
        img_resnet_format = np.moveaxis(img_flipped.astype(np.float32) / 255, -1, 0) # HWC to CHW
        agent_pose = np.concatenate([obs['robot0_eef_pos'][1:3], obs['robot0_gripper_qpos'][:1]])

        velocity_joint = obs['robot0_joint_vel']
        velocity_gripper = obs['robot0_gripper_qvel']
        mag_joint = np.linalg.norm(velocity_joint)
        mag_gripper = np.linalg.norm(velocity_gripper)
        mag_total = mag_joint + 0.1 * mag_gripper

        obs = {
            'step_count': self.timestep,
            'cur_time': self.cur_time,
            'full_image': img_resnet_format,
            'joint_velocity': velocity_joint,
            'joint_position': obs['robot0_joint_pos'],
            'gripper_velocity': velocity_gripper,
            'agent_velocity': mag_total,
            'agent_pose': agent_pose,
            'wait_time': self.wait_timer,
            'wait_times_each_visit': np.pad(
                np.asarray(self.wait_times_each_visit, dtype=np.float32)[:MAX_GOAL_VISITS],
                (0, max(0, MAX_GOAL_VISITS - len(self.wait_times_each_visit))),
                constant_values=-1,
            )
        }
        return obs


    def step_liftqa_state_only(self, obs, reward, done, info):
        self.last_x = obs['robot0_eef_pos'][0]
        self.last_z = obs['robot0_eef_pos'][2]

        outside_goal_threshold = 0.1

        inside_camera_ymin = 0.1395  # prev: 0.1403 
        inside_camera_ymax = 0.2074  # prev: 0.2066
        inside_camera_zmin = 1.090   # prev: 1.099
        inside_camera_zmax = 1.14    # prev: 1.131


        at_camera_y = inside_camera_ymin <= obs['cube_pos'][1] <= inside_camera_ymax
        at_camera_z = inside_camera_zmin <= obs['cube_pos'][2] <= inside_camera_zmax

        outside_camera_y = obs['cube_pos'][1] < inside_camera_ymin - outside_goal_threshold or obs['cube_pos'][1] > inside_camera_ymax + outside_goal_threshold
        outside_camera_z = obs['cube_pos'][2] < inside_camera_zmin - outside_goal_threshold or obs['cube_pos'][2] > inside_camera_zmax + outside_goal_threshold

        inside_goal_area =  at_camera_y and at_camera_z
        outside_goal_area = outside_camera_y or outside_camera_z
        
        at_start_y = -0.3788 <= obs['cube_pos'][1] <= -0.2189
        at_start_z = obs['cube_pos'][2] <= 0.833
        inside_start_area = at_start_y and at_start_z

        if self.goal_stage == 0 and inside_goal_area:
            print(f"Reach the first goal, wait for 2 seconds")
            # move placard into view [-0.15, 0.2, 0.3]
            
            reward = 0.3  # Reached first goal
            self.goal_stage = 1
            self.wait_timer = 0.0  # Start wait timer when entering stage 1
        
        if inside_goal_area:
            self.wait_timer += 1.0 * (1/self.control_freq)  # Accumulate wait time
        
        if self.goal_stage == 1 and self.wait_timer >= self.wait_duration:
            print(f"Wait completed, move to second goal")
            reward = 0.6  # Finished waiting, advance to next stage
            self.goal_stage = 2
        
        if self.goal_stage == 2 and not inside_goal_area:
            self.prevent_wait_timer_first_visit = True

        if self.goal_stage > 0 and inside_start_area and self.robots[0].gripper['right'].current_action[0] < 0: # gripper open
            self.finished_timer += 1.0 * (1/self.control_freq)
            if self.finished_timer >= 2.0: # wait at least 0.2 seconds before finishing
                print(f"Reach the second goal")
                reward = 1 if self.goal_stage == 2 else reward
                done = True
        
        # Generic visit detection tracking
        if inside_goal_area and not self._visit_active:
            self._visit_active = True
            self._current_visit_timer = 0.0  # start a new visit

        # Accumulate while inside (regardless of stage)
        if self._visit_active:
            
            self._current_visit_timer += 1.0 * (1/self.control_freq)
            if self._current_visit_timer > self.wait_duration - 0.3: # like a flashing light
                self.sim.model.body_pos[self.obj_body_id["camera_active_placard"]] = self.table_offset + np.array([-0.15, 0.2, 0.3])
            if self._current_visit_timer >= self.wait_duration:
                self.sim.model.body_pos[self.obj_body_id["camera_active_placard"]] = self.table_offset + np.array([100, 100, 100])  # move out of view

        # Detect exiting goal area
        if self._visit_active and outside_goal_area:
            self.wait_times_each_visit.append(self._current_visit_timer)
            self._current_visit_timer = 0.0
            self._visit_active = False
            self.sim.model.body_pos[self.obj_body_id["camera_active_placard"]] = self.table_offset + np.array([100, 100, 100])  # move out of view

        obs_extra = self._get_obs_custom(obs)
        obs.update(obs_extra)
        return obs, reward, done, info
    

    def step(self, action):
        
        if self.motion_constrained:
            dx = 2 * (self.initial_robot_x - self.last_x) # move back towards initial x position (Proportional control)
            action[0] = dx
            # if self.last_z < self.table_offset[2] + 0.03:
            #     action[2] = 0 # don't allow lowering if close to table
            action[3] = 0.0  # no rotation
            action[4] = 0.0
            action[5] = 0.0

        obs, reward, done, info = super().step(action)

        
        obs, reward, done, info = self.step_liftqa_state_only(obs, reward, done, info)

        return obs, reward, done, info
    
    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04

def translate_3_dim_action_to_7d(action_3d):
    dy, dz, gripper = action_3d
    dx = 0.0
    ax = 0.0
    ay = 0.0
    az = 0.0
    return np.array([dx, dy, dz, ax, ay, az, gripper])

def create_env(
        render: bool = True,
        constrain_motion: bool = True,
        allow_camera_control: bool = False,
        control_hz: int = 10,
        max_episode_length: int = 1000,
        seed: int = 0,
        camera_height_width: tuple = (144, 144),
        use_joint_control: bool = False,
    ) -> LiftQA:
        

        controller_config = {}
        
        # if use_ik:
        #     from robosuite.controllers import load_composite_controller_config
        #     controller_config = load_composite_controller_config(controller="WHOLE_BODY_IK")
        #     controller_config["composite_controller_specific_configs"]["ik_input_ref_frame"] = "world"

        controller_str = "JOINT_POSITION" if use_joint_control else "OSC_POSE"

        # load default controller config for robot
        controller_config = {
            "body_parts": {
                "right": load_part_controller_config(default_controller=controller_str)
            },
        }
        controller_config["body_parts"]["right"]["gripper"] = {"type": "GRIP"}
        controller_config["body_parts"]["right"]["kp"] = 800 # default kp is 150 increase stiffness for better teleop control
        controller_config["body_parts"]["right"]["kp_limits"] = [0, 1000] # default kp is 150 increase stiffness for better teleop control
        # controller_config["body_parts"]["right"]["damping_ratio"] = 8
        # controller_config["body_parts"]["right"]["damping_ratio_limits"] = [0, 10]
        controller_config["body_parts"]["right"]["ramp_ratio"] = 0.5

        init_jpos = np.array([-0.7414455 , -1.04108567,  1.83158798, -2.36423285, -1.57200527, -2.26416281])

        return LiftQA(
            robots="UR5e",
            controller_configs=controller_config,
            has_renderer=render,
            initialization_noise=None,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            render_camera="agentview" if not allow_camera_control else None, # "agentview", # None, # "agentview", # None or "agentview", frontview
            # renderer="mujoco", # mjviewer always hard resets
            camera_heights=camera_height_width[0],
            camera_widths=camera_height_width[1],
            initial_robot_joints=init_jpos,
            horizon=max_episode_length,
            control_freq=control_hz,
            seed=seed,
            hard_reset=False,
            constrain_motion=constrain_motion
        )

if __name__ == "__main__":

    env = LiftQA(
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon=10000,
        control_freq=10,
        render_camera=None,
    )

    env.reset()
    env.render()

    while True:
        env.step(np.zeros(env.action_dim))
        env.render()

