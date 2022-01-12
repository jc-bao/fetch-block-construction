import os
from fetch_block_construction.envs.robotics import utils
from fetch_block_construction.envs.robotics import fetch_env, rotations
import numpy as np
import mujoco_py
from .xml import generate_xml
import tempfile


class FetchTower(fetch_env.FetchEnv):
    def __init__(self,num_blocks=1):
        self.total_num_blocks = 6
        self.num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.total_num_blocks)]
        with tempfile.NamedTemporaryFile(mode='wt', dir=F"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/assets/fetch/", delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.total_num_blocks))
            MODEL_XML_PATH = fp.name
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(self.total_num_blocks):
            initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, obs_type='dict', render_size=42, reward_type='sparse')
        os.remove(MODEL_XML_PATH)
        self._max_episode_steps = 50*self.num_blocks

    def gripper_pos_far_from_goals(self, gripper_pos, goal):
        distances = np.linalg.norm(goal.reshape(-1,3) - gripper_pos, axis=-1)
        return np.all(distances > self.distance_threshold * 2)

    def compute_reward(self, achieved_goal, goal, info):
        subgoal_distances = np.linalg.norm(achieved_goal.reshape(-1,3) - goal.reshape(-1,3), axis=-1)
        reward = -np.sum((subgoal_distances > self.distance_threshold).astype(np.float32), axis=0)
        reward = np.asarray(reward)
        gripper_pos = info['gripper_pos']
        np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(gripper_pos, goal))
        return reward
        
    def _get_obs(self):
        # robot
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        obs = np.concatenate([ 
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])
        # task
        achieved_goal = []
        for i in range(self.num_blocks):
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp
            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])
            achieved_goal = np.append(achieved_goal, object_i_pos.copy())
        return_dict = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'info': {'gripper_pos': grip_pos},
        }
        return return_dict

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for i in range(int(len(self.goal)/3)):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.goal[i * 3:(i + 1) * 3] - sites_offset[i]
        self.sim.forward()

    def _reset_sim(self):
        assert self.num_blocks <= 17 # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)
        # Randomize start position of objects.
        prev_obj_xpos = []
        for i, obj_name in enumerate(self.object_names):
            if i < self.num_blocks:
                object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                while not ((np.linalg.norm(object_xypos - self.initial_gripper_xpos[:2]) >= 0.1) and np.all([np.linalg.norm(object_xypos - other_xpos) >= 0.06 for other_xpos in prev_obj_xpos])):
                    object_xypos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                prev_obj_xpos.append(object_xypos)
                object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xypos
                object_qpos[2] = self.height_offset
            else:
                object_qpos = [0.5, 0.4, i*.06-0.1, 1., 0., 0., 0.]
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
        self.sim.forward()
        return True

    def _sample_goal(self):
        goals = []
        # sample goal0
        goal_object0 = self.initial_gripper_xpos[:3] + \
            self.np_random.uniform(-self.target_range, self.target_range,size=3)
        goal_object0 += self.target_offset
        goal_object0[2] = self.height_offset
        if self.np_random.uniform() < 0.5 and self.num_blocks==1:
            goal_object0[2] += self.np_random.uniform(0, 0.45)
        goals.append(goal_object0)
        # sample other goals
        for i in range(1, self.num_blocks):
            goal_objecti = goal_object0.copy()
            goal_objecti[-1] += 0.05*i
            goals.append(goal_objecti)
        return np.concatenate(goals, axis=0).copy()

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = np.linalg.norm(achieved_goal.reshape(-1,3) - desired_goal.reshape(-1,3), axis=-1)
        return (subgoal_distances < self.distance_threshold).all()

    def _set_action(self, action):
        assert action.shape == (4,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(F"action {action}")
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'gripper_pos': self.sim.data.get_site_xpos('robot0:grip')
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def change(self, config):
        self.num_blocks = int(config)
        self._max_episode_steps = 50*self.num_blocks