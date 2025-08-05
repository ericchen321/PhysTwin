# Author: Ganidhu
import numpy as np
import warp as wp
from qqtt.model.diff_simulator.spring_mass_hybrid import MassSpringIntegrator

wp.init()
wp.set_device("cuda:0")
wp.config.verify_autograd_array_access = True

def build_test_system():
    # build simple mass spring system
    x = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    v = np.zeros((1, 4))
    v_before_collision = v
    v_before_ground = v

    vertice_forces = np.zeros((1, 4))
    object_collision_flag = False
    num_object_points = 4
    masses = np.ones((1, 4))
    dt = 0.01
    drag_damping = 0
    reverse_factor = 0
    collide_elas = 0
    collide_fric = 0
    collision_dist = 0.02
    collision_indices = np.zeros((v.shape[0], 500))
    collision_number = np.zeros(v.shape[0])
    collide_object_elas = 0.7
    collide_object_fric = 0.3
    masks = None
    current_object_points = 
    current_object_visibilities
    current_object_motions_valid
    num_valid_visibilities
    num_valid_motions
    prev_acc
    acc_count
    cfg
    num_original_points
    neigh_indices
    num_surface_points
    num_substeps


def check_forward_prop(system):
    x = wp.to_torch(system["wp_x"])
    v = wp.to_torch(system["wp_v"])
    v_before_collision = wp.to_torch(system["wp_v_before_collision"])
    v_before_ground = wp.to_torch(system["wp_v_before_ground"])
    vertice_forces = wp.to_torch(system["wp_vertice_forces"])
    v_initial = system["wp_v"] # TODO: for multi-step systems this would be a different value.
    object_collision_flag = system["object_collision_flag"]
    num_object_points = system["num_object_points"]
    masses = system["wp_masses"]
    dt = system["dt"]
    drag_damping = system["drag_damping"]
    reverse_factor = system["reverse_factor"]
    collide_elas = system["wp_collide_elas"]
    collide_fric = system["wp_collide_fric"]
    collision_dist = system["collision_dist"]
    collision_indices = 

def check_backward_prop(system):
    pass

def main():
    system_state = build_test_system()

    check_forward_prop(system_state)

    check_backward_prop(system_state)


if __name__ == "__main__":
    main()