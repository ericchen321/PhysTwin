# Author: Ganidhu
import numpy as np
import warp as wp
from qqtt.model.diff_simulator.spring_mass_hybrid import MassSpringHybridIntegrator

wp.init()
wp.set_device("cuda:0")
wp.config.verify_autograd_array_access = True

def build_test_system():
    # build simple mass spring system
    x = np.array([[0, 1, 0]])
    wp_x = wp.from_numpy(x, requires_grad=True, device="cuda:0")
    wp_v = wp.zeros((1, 1))
    wp_v_before_collision = wp_v
    wp_v_before_ground = wp_v

    wp_vertice_forces = wp.zeros((1, 1))
    object_collision_flag = False
    num_object_points = 4
    wp_masses = wp.ones((1, 1))
    dt = 0.01
    drag_damping = wp.zeros(1)
    reverse_factor = wp.zeros(1)
    collide_elas = wp.zeros(1)
    collide_fric = wp.zeros(1)

    state = {
        "wp_x": wp_x,
        "wp_v": wp_v,
        "wp_v_before_collision": wp_v_before_collision,
        "wp_v_before_ground": wp_v_before_ground,
        "wp_vertice_forces": wp_vertice_forces,
        "wp_masses": wp_masses,
        "dt": dt,
        "drag_damping": drag_damping,
        "reverse_factor": reverse_factor,
        "wp_collide_elas": collide_elas,
        "wp_collide_fric": collide_fric
    }

    return state


def check_forward_prop(system):
    x = wp.to_torch(system["wp_x"])
    v = wp.to_torch(system["wp_v"])
    v_before_collision = wp.to_torch(system["wp_v_before_collision"])
    v_before_ground = wp.to_torch(system["wp_v_before_ground"])
    vertice_forces = wp.to_torch(system["wp_vertice_forces"])
    num_object_points = system["num_object_points"]
    masses = system["wp_masses"]
    dt = system["dt"]
    drag_damping = system["drag_damping"]
    reverse_factor = system["reverse_factor"]
    collide_elas = system["wp_collide_elas"]
    collide_fric = system["wp_collide_fric"]
    num_substeps = 5

    tape = wp.Tape()

    with tape:
        t_x, t_v, t_v_before_collision, t_v_before_ground, t_vertice_forces = MassSpringHybridIntegrator.apply(
            x,
            v,
            v_before_collision,
            v_before_ground,
            vertice_forces,
            num_object_points,
            masses,
            drag_damping,
            collide_elas,
            collide_fric,
            reverse_factor,
            dt,
            tape,
            num_substeps
        )

        print(t_vertice_forces)

def check_backward_prop(system):
    pass

def main():
    system_state = build_test_system()

    check_forward_prop(system_state)

    check_backward_prop(system_state)


if __name__ == "__main__":
    main()