import torch
from qqtt.utils import logger, cfg
import warp as wp

from qqtt.model.diff_simulator.kernels import (
    update_vel_from_force,
    object_collision,
    integrate_ground_collision,
    compute_distances,
    compute_neigh_indices,
    compute_chamfer_loss,
    compute_track_loss,
    compute_acc_loss,
    compute_final_loss,
    set_int,
    copy_vec3,
    update_potential_collision,
    update_acc,
    copy_int,
    copy_float
)

from qqtt.model.diff_simulator.spring_mass_warp import (
    State
)

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True

def gather_spring_endpoints(x, control_x, num_object_points, springs):
    num_springs = springs.shape[0]
    device = x.device
    dtype = x.dtype

    # Preallocate just the final needed shape
    spring_pos = torch.empty((num_springs, 2, 3), device=device, dtype=dtype)

    idx1 = springs[:, 0]
    idx2 = springs[:, 1]

    is_obj1 = idx1 < num_object_points
    is_obj2 = idx2 < num_object_points

    assert torch.all((idx1[is_obj1] >= 0) & (idx1[is_obj1] < num_object_points)), \
        f"Invalid spring index found: {idx1[(idx1 < 0) | (idx1 >= num_object_points)]}"


    # First endpoint
    spring_pos[is_obj1, 0] = x[idx1[is_obj1]]
    spring_pos[~is_obj1, 0] = control_x[idx1[~is_obj1] - num_object_points]

    # Second endpoint
    spring_pos[is_obj2, 1] = x[idx2[is_obj2]]
    spring_pos[~is_obj2, 1] = control_x[idx2[~is_obj2] - num_object_points]
    
    return spring_pos

def eval_springs_energy(
    x: torch.Tensor,
    control_x: torch.Tensor,
    dx: torch.Tensor,
    springs: torch.Tensor,
    spring_l0s: torch.Tensor,
    num_springs: int,
    spring_k: float,
    dashpot_damping: float
):
    t_spring_k = torch.tensor([spring_k], device="cuda")
    t_spring_b = torch.tensor([dashpot_damping], device="cuda")

    num_samples, num_particles, _ = x.shape
    epsi_ks = 1e-6
    epsi_bs = 1e-6

    # expand spring k and b, and apply positivity func
    spring_ks = torch.broadcast_to(
        t_spring_k.view(1, 1, 1),
        (1, num_springs, 1))
    positive_fun = torch.nn.ReLU().to("cuda")
    spring_ks_pos = positive_fun(spring_ks) + epsi_ks
    spring_bs = torch.broadcast_to(
        t_spring_b.view(1, 1, 1),
        (1, num_springs, 1))
    spring_bs_pos = positive_fun(spring_bs) + epsi_bs

    # extract end point positions of each spring
    spring_pos = gather_spring_endpoints(x, control_x, num_particles, springs)  # (1, num_springs, 2, 3)
    
    # compute each spring's direction vec
    d = spring_pos[:, :, 1] - spring_pos[:, :, 0]
    d = d.view(num_samples, num_springs, 3)

    # compute each spring's length
    spring_l = torch.linalg.vector_norm(
        d, dim=-1).view(num_samples, num_springs, 1)

    # compute elastic energy
    l_m_l0 = spring_l - spring_l0s.view(1, num_springs, 1)
    e_elastic = 0.5 * spring_ks_pos * (l_m_l0 ** 2)

    # compute strain
    strain = l_m_l0 / (spring_l0s.view(1, num_springs, 1) + 1e-6)

    # compute relative velocity betwen endpoints
    spring_vel = dx[:, springs].view(
        1, num_springs, 2, 3)
    vrel = spring_vel[:, :, 1] - spring_vel[:, :, 0]
    vrel = vrel.view(1, num_springs, 3)


    # compute velocity along spring direction
    d_unit = d / (torch.linalg.vector_norm(
        d, dim=-1, keepdim=True) + 1e-6)
    vrel_proj_speed = torch.sum(
        vrel * d_unit, dim=-1, keepdim=True).view(
            1, num_springs, 1)
    vrel_proj = vrel_proj_speed * d_unit
    vrel_proj = vrel_proj.view(1, num_springs, 3)

    # compute strain rate as follows: first, compute grad of
    # of strain wrt relative pos d; then dot product with the
    # relative velocity along spring direction
    dstrain_dd = torch.autograd.grad(
        strain,
        d,
        grad_outputs=torch.ones_like(strain),
        create_graph=True, retain_graph=True)[0]
    dstrain_dd = dstrain_dd.view(1, num_springs, 3)
    strain_rate = torch.sum(
        dstrain_dd * vrel_proj, dim=-1, keepdim=True).view(
            1, num_springs, 1)
    
    spring_bs_pos = positive_fun(
        spring_bs.view(1, num_springs, 1)) + epsi_bs
    e_damping = 0.5 * spring_bs_pos * (strain_rate ** 2)
    e_damping = e_damping.view(1, num_springs, 1)

    return {
        "energy_elastic": e_elastic,
        "energy_damping": e_damping
    }


class MassSpringLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_curr: torch.Tensor,
        v_curr: torch.Tensor,
        v_initial: torch.Tensor,
        num_original_points: int,
        num_object_points: int,
        num_surface_points: int,
        num_valid_visibilities: float,
        track_weight: float,
        chamfer_weight: float,
        acc_weight: float,
        prev_acc: torch.Tensor,
        acc_count: float,
        num_valid_motions: int,
        current_object_points: wp.array,
        current_object_visibilities: wp.array,
        current_object_motions_valid: wp.array
    ):
        # Initialize loss tensors
        chamfer_loss = torch.zeros(1, device=x_curr.device, requires_grad=True)
        track_loss = torch.zeros(1, device=x_curr.device, requires_grad=True)
        acc_loss = torch.zeros(1, device=x_curr.device, requires_grad=True)
        loss = torch.zeros(1, device=x_curr.device, requires_grad=True)

        wp_x_curr = wp.from_torch(x_curr, requires_grad=True)
        wp_v_curr = wp.from_torch(v_curr, requires_grad=True)
        wp_v_initial = wp.from_torch(v_initial, requires_grad=True)

        # Create warp tensors for loss components
        wp_chamfer_loss = wp.from_torch(chamfer_loss, requires_grad=True)
        wp_track_loss = wp.from_torch(track_loss, requires_grad=True)
        wp_acc_loss = wp.from_torch(acc_loss, requires_grad=True)
        wp_loss = wp.from_torch(loss, requires_grad=True)

        distance_matrix = wp.zeros(
            (num_original_points, num_surface_points), requires_grad=False
        )
        neigh_indices = wp.zeros(
            (num_original_points), dtype=wp.int32, requires_grad=False
        )

        # Create tape for gradient computation
        ctx.tape = wp.Tape()
        with ctx.tape:
            # Compute the chamfer loss
            wp.launch(
                compute_distances,
                dim=(num_original_points, num_surface_points),
                inputs=[
                    wp_x_curr,
                    current_object_points,
                    current_object_visibilities,
                ],
                outputs=[distance_matrix],
            )

            wp.launch(
                compute_neigh_indices,
                dim=num_original_points,
                inputs=[distance_matrix],
                outputs=[neigh_indices],
            )

            wp.launch(
                compute_chamfer_loss,
                dim=num_original_points,
                inputs=[
                    wp_x_curr,
                    current_object_points,
                    current_object_visibilities,
                    num_valid_visibilities,
                    neigh_indices,
                    chamfer_weight,
                ],
                outputs=[wp_chamfer_loss],
            )

            # Compute the tracking loss
            wp.launch(
                compute_track_loss,
                dim=num_original_points,
                inputs=[
                    wp_x_curr,
                    current_object_points,
                    current_object_motions_valid,
                    num_valid_motions,
                    track_weight,
                ],
                outputs=[wp_track_loss],
            )

            # Compute acceleration loss
            wp.launch(
                compute_acc_loss,
                dim=num_object_points,
                inputs=[
                    wp_v_initial,
                    wp_v_curr,
                    prev_acc,
                    num_object_points,
                    acc_count,
                    acc_weight,
                ],
                outputs=[wp_acc_loss],
            )

            # Compute final combined loss
            wp.launch(
                compute_final_loss,
                dim=1,
                inputs=[wp_chamfer_loss, wp_track_loss, wp_acc_loss],
                outputs=[wp_loss],
            )

        # Convert back to torch
        final_loss = wp.to_torch(wp_loss)
        
        # Save context for backward pass
        ctx.save_for_backward(x_curr, v_curr, v_initial)
        ctx.wp_x_curr = wp_x_curr
        ctx.wp_v_curr = wp_v_curr
        ctx.wp_v_initial = wp_v_initial
        ctx.wp_loss = wp_loss

        return final_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implement backward pass for MassSpringLoss
        """
        x_curr, v_curr, v_initial = ctx.saved_tensors
        
        # Set the gradient of the loss to the incoming gradient
        ctx.wp_loss.grad = wp.from_torch(grad_output)
        
        # Run backward pass through the tape
        ctx.tape.backward(loss=ctx.wp_loss)
        
        # Extract gradients
        grad_x_curr = None
        grad_v_curr = None
        grad_v_initial = None
        
        if ctx.wp_x_curr.grad is not None:
            grad_x_curr = wp.to_torch(ctx.wp_x_curr.grad)
        
        if ctx.wp_v_curr.grad is not None:
            grad_v_curr = wp.to_torch(ctx.wp_v_curr.grad)

        # Return gradients for all inputs (None for non-tensor inputs)
        return (
            grad_x_curr,      # x_curr
            grad_v_curr,      # v_curr
            None,             # v_initial
            None,             # num_original_points
            None,             # num_object_points
            None,             # num_surface_points
            None,             # num_valid_visibilities
            None,             # track_weight
            None,             # chamfer_weight
            None,             # acc_weight
            None,             # prev_acc
            None,             # acc_count
            None,             # num_valid_motions
            None,             # current_object_points
            None,             # current_object_visibilities
            None,             # current_object_motions_valid
        )

class MassSpringHybridIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        v: torch.Tensor,
        v_before_collision: torch.Tensor,
        v_before_ground: torch.Tensor,
        vertice_forces: torch.Tensor,
        num_object_points: int,
        control_x: torch.Tensor,
        wp_masses: wp.array,
        drag_damping: float,
        wp_collide_elas: wp.array,
        wp_collide_fric: wp.array,
        reverse_factor: float,
        dt: float,
        num_substeps: int,
        springs: torch.Tensor,
        spring_l0s: torch.Tensor,
        spring_k: float,
        dashpot_damping: float
    ):
        # TODO: enable object collision
        object_collision_flag = False

        # Compute substep dt
        substep_dt = dt / num_substeps
        
        # Convert initial state to Warp arrays
        wp_v_before_collision = wp.from_torch(v_before_collision, dtype=wp.vec3, requires_grad=True)
        wp_v_before_ground = wp.from_torch(v_before_ground, dtype=wp.vec3, requires_grad=True)
        wp_x = wp.from_torch(x, dtype=wp.vec3, requires_grad=True)
        wp_v = wp.from_torch(v, dtype=wp.vec3, requires_grad=True)

        # Create working copies for substep integration
        current_x_torch = x.clone().requires_grad_(True)
        current_v_torch = v.clone().requires_grad_(True)
        current_x = wp.clone(wp_x)
        current_v = wp.clone(wp_v)

        ctx.tape = wp.Tape()
        with ctx.tape:
            # Run multiple substeps
            for substep in range(num_substeps):
                # Recompute forces at current position and velocity
                current_x_torch = wp.to_torch(current_x).requires_grad_(True)
                current_v_torch = wp.to_torch(current_v).requires_grad_(True)
                
                # Evaluate spring energies at current state
                elastic_energy, damping_energy = eval_springs_energy(
                    current_x_torch.unsqueeze(0),
                    control_x, 
                    current_v_torch.unsqueeze(0),
                    springs,
                    spring_l0s,
                    springs.shape[0],
                    spring_k,
                    dashpot_damping
                )
                
                # Compute forces via autodiff
                elastic_force = -torch.autograd.grad(
                    elastic_energy, current_x_torch, create_graph=True, retain_graph=True
                )[0]
                damping_force = -torch.autograd.grad(
                    damping_energy, current_v_torch, create_graph=True, retain_graph=True
                )[0]

                total_forces = elastic_force + damping_force
                wp_total_forces = wp.from_torch(total_forces, dtype=wp.vec3)

                # Determine which velocity array to use based on collision flag
                if object_collision_flag:
                    output_v = wp.clone(wp_v_before_collision)
                else:
                    output_v = wp.clone(wp_v_before_ground)

                # Update velocity using recomputed forces
                wp.launch(
                    kernel=update_vel_from_force,
                    dim=num_object_points,
                    inputs=[
                        current_v,
                        wp_total_forces,
                        wp_masses,
                        substep_dt,
                        drag_damping,
                        reverse_factor,
                    ],
                    outputs=[output_v],
                )

                integration_v = output_v
                
                # Integrate position and velocity for this substep
                next_x = wp.zeros_like(current_x, requires_grad=True)
                next_v = wp.zeros_like(current_v, requires_grad=True)
                wp.launch(
                    kernel=integrate_ground_collision,
                    dim=num_object_points,
                    inputs=[
                        current_x,
                        integration_v,
                        wp_collide_elas,
                        wp_collide_fric,
                        substep_dt,
                        reverse_factor,
                    ],
                    outputs=[next_x, next_v],
                )
                
                # Update current state for next substep
                current_x = next_x
                current_v = next_v

        # Convert final results back to torch
        torch_x = wp.to_torch(current_x)
        torch_v = wp.to_torch(current_v)
        torch_v_before_collision = wp.to_torch(wp_v_before_collision)
        torch_v_before_ground = wp.to_torch(wp_v_before_ground)
        final_forces = elastic_force + damping_force  # Use the last computed forces

        # Save for backward pass
        ctx.save_for_backward(x, v, v_before_collision, v_before_ground)
        
        # Store warp arrays and other info needed for backward
        ctx.wp_x = wp_x
        ctx.wp_v = wp_v
        ctx.wp_v_before_collision = wp_v_before_collision
        ctx.wp_v_before_ground = wp_v_before_ground
        ctx.final_x = current_x
        ctx.final_v = current_v

        return torch_x, torch_v, torch_v_before_collision, torch_v_before_ground, final_forces

    @staticmethod
    def backward(ctx, grad_x, grad_v, grad_v_before_collision, grad_v_before_ground, grad_forces):
        """
        Fixed backward pass for MassSpringHybridIntegrator
        """
        x, v, v_before_collision, v_before_ground = ctx.saved_tensors

        # Set gradients on the final outputs
        if grad_x is not None:
            ctx.final_x.grad = wp.from_torch(grad_x, dtype=wp.vec3)
        if grad_v is not None:
            ctx.final_v.grad = wp.from_torch(grad_v, dtype=wp.vec3)
        if grad_v_before_collision is not None and ctx.wp_v_before_collision.grad is not None:
            ctx.wp_v_before_collision.grad = wp.from_torch(grad_v_before_collision, dtype=wp.vec3)
        if grad_v_before_ground is not None and ctx.wp_v_before_ground.grad is not None:
            ctx.wp_v_before_ground.grad = wp.from_torch(grad_v_before_ground, dtype=wp.vec3)

        # Run backward pass through the computational graph
        ctx.tape.backward()

        # Extract gradients w.r.t. inputs
        grad_input_x = None
        grad_input_v = None
        grad_input_v_before_collision = None
        grad_input_v_before_ground = None

        if ctx.wp_x.grad is not None:
            grad_input_x = wp.to_torch(ctx.wp_x.grad)
        if ctx.wp_v.grad is not None:
            grad_input_v = wp.to_torch(ctx.wp_v.grad)
        if ctx.wp_v_before_collision.grad is not None:
            grad_input_v_before_collision = wp.to_torch(ctx.wp_v_before_collision.grad)
        if ctx.wp_v_before_ground.grad is not None:
            grad_input_v_before_ground = wp.to_torch(ctx.wp_v_before_ground.grad)

        return (
            grad_input_x,                    # x
            grad_input_v,                    # v
            grad_input_v_before_collision,   # v_before_collision
            grad_input_v_before_ground,      # v_before_ground
            grad_forces,                     # vertice_forces (pass through)
            None,                           # num_object_points
            None,                           # wp_masses
            None,                           # drag_damping
            None,                           # wp_collide_elas
            None,                           # wp_collide_fric
            None,                           # reverse_factor
            None,                           # dt
            None,                           # num_substeps
            None,                           # springs
            None,                           # spring_l0s
            None,                           # spring_k
            None,                           # dashpot_damping
        )
    
class SpringMassSystemHybrid:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # Store spring parameters for hybrid integrator
        self.springs = init_springs
        self.spring_l0s = init_rest_lengths
        self.spring_k = spring_Y  # Using spring_Y as spring_k

        # Store masses and collision parameters
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

        # Initialize collision parameters
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Current state for step and loss computation
        self.current_x = None
        self.current_v = None
        self.current_v_before_collision = None
        self.current_v_before_ground = None
        self.current_forces = None
        self.current_loss = None

        # Initialize with starting state
        self.reset_to_initial_state()

        # Note: Removed CUDA graph creation since we're using PyTorch autograd functions
        # The hybrid functions handle their own gradient computation

    def reset_to_initial_state(self):
        """Reset to initial state"""
        self.current_x = wp.to_torch(self.wp_init_vertices).clone().requires_grad_(True)
        self.current_v = wp.to_torch(self.wp_init_velocities).clone().requires_grad_(True)
        self.current_v_before_collision = self.current_v.clone()
        self.current_v_before_ground = self.current_v.clone()
        self.current_forces = torch.zeros_like(self.current_x)

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        """Set initial state from warp arrays"""
        assert (
            self.num_object_points == wp_x.shape[0]
        )

        # Convert warp arrays to torch tensors
        self.current_x = wp.to_torch(wp_x).clone().requires_grad_(True)
        self.current_v = wp.to_torch(wp_v).clone().requires_grad_(True)
        self.current_v_before_collision = self.current_v.clone()
        self.current_v_before_ground = self.current_v.clone()

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        """Update acceleration for loss computation"""
        if cfg.data_type == "real":
            wp.launch(
                update_acc,
                dim=self.num_object_points,
                inputs=[
                    wp.from_torch(self.current_v.detach()),
                    wp.from_torch(self.current_v.detach()),  # Will be updated after step
                ],
                outputs=[self.prev_acc],
            )

    def update_collision_graph(self):
        assert self.object_collision_flag
        wp_x = wp.from_torch(self.current_x.detach(), dtype=wp.vec3)
        self.collision_grid.build(wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        """Use hybrid integrator to compute next simulation state"""
        # Use the hybrid integrator
        (
            self.current_x,
            self.current_v,
            self.current_v_before_collision,
            self.current_v_before_ground,
            self.current_forces
        ) = MassSpringHybridIntegrator.apply(
            self.current_x,
            self.current_v,
            self.current_v_before_collision,
            self.current_v_before_ground,
            self.current_forces,
            self.num_object_points,
            self.controller_points,
            self.wp_masses,
            self.drag_damping,
            self.wp_collide_elas,
            self.wp_collide_fric,
            self.reverse_factor,
            self.dt,
            self.num_substeps,
            self.springs,
            self.spring_l0s,
            self.spring_k,
            self.dashpot_damping
        )

    def calculate_loss(self):
        """Use hybrid loss function to compute loss"""
        if cfg.data_type == "real":
            # Get initial velocity for acceleration loss
            v_initial = wp.to_torch(self.wp_init_velocities).requires_grad_(True)
            
            # Compute acceleration count
            acc_count_value = float(wp.to_torch(self.acc_count).item())
            
            self.current_loss = MassSpringLoss.apply(
                self.current_x,
                self.current_v,
                v_initial,
                self.num_original_points,
                self.num_object_points,
                self.num_surface_points,
                float(self.num_valid_visibilities),
                cfg.track_weight,
                cfg.chamfer_weight,
                cfg.acc_weight,
                self.prev_acc,
                acc_count_value,
                self.num_valid_motions,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.wp_current_object_motions_valid
            )
        else:
            # For synthetic data, compute simple L2 loss
            gt_points = wp.to_torch(self.wp_current_object_points)
            self.current_loss = torch.mean(torch.sum((self.current_x - gt_points) ** 2, dim=-1))

    def calculate_simple_loss(self):
        """Calculate simple L2 loss for synthetic data"""
        gt_points = wp.to_torch(self.wp_current_object_points)
        self.current_loss = torch.mean(torch.sum((self.current_x - gt_points) ** 2, dim=-1))

    def clear_loss(self):
        """Clear current loss"""
        self.current_loss = None

    # Functions used to load the parameters
    def set_spring_Y(self, spring_Y):
        """Set spring stiffness parameter"""
        self.spring_k = spring_Y.item() if hasattr(spring_Y, 'item') else float(spring_Y)

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )

    def get_current_state(self):
        """Get current simulation state as torch tensors"""
        return self.current_x, self.current_v

    def get_current_loss(self):
        """Get current loss value"""
        return self.current_loss