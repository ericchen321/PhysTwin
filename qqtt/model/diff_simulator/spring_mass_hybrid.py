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

def eval_springs_energy(
    x: torch.Tensor,
    dx: torch.Tensor,
    springs: torch.Tensor,
    num_springs: float,
    spring_k: float,
    dashpot_damping: float
):
    t_spring_k = torch.Tensor([spring_k], device="cuda")
    t_spring_b = torch.Tensor([dashpot_damping], device="cuda")

    # expand spring k and b, and apply positivity func
    spring_ks = torch.broadcast_to(
        t_spring_k.view(1, 1, 1),
        (1, num_springs, 1))
    # spring_ks_pos = self.param_pos_fun(spring_ks) + self.epsi_ks
    spring_bs = torch.broadcast_to(
        t_spring_b.view(1, 1, 1),
        (1, num_springs, 1))
    # spring_bs_pos = self.param_pos_fun(spring_bs) + self.epsi_bs

    # extract end point positions of each spring
    x = dict_in["x"].clone()
    dx = dict_in["dx"].clone()
    spring_pos = x[:, self.springs].view(
        num_samples, self.num_springs, 2, 3)
    
    # compute each spring's direction vec
    d = spring_pos[:, :, 1] - spring_pos[:, :, 0]
    d = d.view(num_samples, self.num_springs, 3)

    # compute each spring's length
    spring_l = torch.linalg.vector_norm(
        d, dim=-1).view(num_samples, self.num_springs, 1)

    # compute elastic energy
    l_m_l0 = spring_l - self.spring_l0s.view(1, self.num_springs, 1)
    e_elastic = 0.5 * spring_ks_pos * (l_m_l0 ** 2)

    # compute strain
    strain = l_m_l0 / (self.spring_l0s.view(1, self.num_springs, 1) + 1e-6)

    # compute relative velocity betwen endpoints
    spring_vel = dx[:, self.springs].view(
        1, self.num_springs, 2, 3)
    vrel = spring_vel[:, :, 1] - spring_vel[:, :, 0]
    vrel = vrel.view(1, self.num_springs, 3)


    # compute velocity along spring direction
    d_unit = d / (torch.linalg.vector_norm(
        d, dim=-1, keepdim=True) + 1e-6)
    vrel_proj_speed = torch.sum(
        vrel * d_unit, dim=-1, keepdim=True).view(
            1, self.num_springs, 1)
    vrel_proj = vrel_proj_speed * d_unit
    vrel_proj = vrel_proj.view(1, self.num_springs, 3)

    # compute strain rate as follows: first, compute grad of
    # of strain wrt relative pos d; then dot product with the
    # relative velocity along spring direction
    dstrain_dd = torch.autograd.grad(
        strain,
        d,
        grad_outputs=torch.ones_like(strain),
        create_graph=True, retain_graph=True)[0]
    dstrain_dd = dstrain_dd.view(1, self.num_springs, 3)
    strain_rate = torch.sum(
        dstrain_dd * vrel_proj, dim=-1, keepdim=True).view(
            1, self.num_springs, 1)
    
    spring_bs_pos = self.param_pos_fun(
        self.spring_bs.view(1, self.num_springs, 1)) + self.epsi_bs
    e_damping = 0.5 * spring_bs_pos * (strain_rate ** 2)
    e_damping = e_damping.view(1, self.num_springs, 1)


# REQUIRED: tape must be recording the call to MassSpringHybridIntegrator.apply()
class MassSpringHybridIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        v: torch.Tensor,
        v_before_collision: torch.Tensor,
        v_before_ground: torch.Tensor,
        vertice_forces: torch.Tensor,
        num_object_points,
        wp_masses,
        drag_damping,
        wp_collide_elas,
        wp_collide_fric,
        reverse_factor,
        dt,
        tape,
        num_substeps
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
        current_x_torch = x.clone()
        current_v_torch = v.clone()
        current_x = wp.clone(wp_x)
        current_v = wp.clone(wp_v)

        ctx.tape = tape
        # Run multiple substeps
        for substep in range(num_substeps):
            # Recompute forces at current position and velocity
            current_x_torch = wp.to_torch(current_x)
            current_v_torch = wp.to_torch(current_v)
            
            # Make sure gradients are enabled for force computation
            current_x_torch.requires_grad_(True)
            current_v_torch.requires_grad_(True)
            
            # Evaluate spring energies at current state
            elastic_energy, damping_energy = eval_springs_energy(current_x_torch, current_v_torch)
            
            # Compute forces via autodiff
            elastic_force = -torch.autograd.grad(
                elastic_energy, current_x_torch, create_graph=True, retain_graph=True
            )[0]
            damping_force = -torch.autograd.grad(
                damping_energy, current_v_torch, create_graph=True, retain_graph=True
            )[0]

            vertice_forces = elastic_force + damping_force
            wp_vertice_forces = wp.from_torch(vertice_forces, dtype=wp.vec3)

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
                    wp_vertice_forces,
                    wp_masses,
                    substep_dt,
                    drag_damping,
                    reverse_factor,
                ],
                outputs=[output_v],
            )

            # Handle object collision if enabled
            if object_collision_flag:
                # temp_v_before_ground = wp.clone(wp_v_before_ground)
                # wp.launch(
                #     kernel=object_collision,
                #     dim=num_object_points,
                #     inputs=[
                #         current_x,
                #         output_v,  # This was wp_v_before_collision
                #         wp_masses,
                #         wp_masks,
                #         wp_collide_object_elas,
                #         wp_collide_object_fric,
                #         collision_dist,
                #         wp_collision_indices,
                #         wp_collision_number,
                #     ],
                #     outputs=[temp_v_before_ground],
                # )
                # Use the collision-processed velocity for integration
                # integration_v = temp_v_before_ground
                pass
            else:
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

        torch_x = wp.to_torch(current_x)
        torch_v = wp.to_torch(current_v)
        torch_v_before_collision = wp.to_torch(wp_v_before_collision)
        torch_v_before_ground = wp.to_torch(wp_v_before_ground)
        torch_vertice_forces = wp.to_torch(wp_vertice_forces)

        # Save for backward pass
        ctx.save_for_backward(
            x,  # Initial position
            v,   # Initial velocity
            torch_vertice_forces # Overall force
        )
        
        # Store warp arrays needed for backward
        ctx.wp_v_before_collision = wp_v_before_collision
        ctx.wp_v_before_ground = wp_v_before_ground
        ctx.wp_vertice_forces = wp_vertice_forces

        # Return final state after all substeps and the loss
        return torch_x, torch_v, torch_v_before_collision, torch_v_before_ground, torch_vertice_forces

        

    def backward(ctx, *grad_output):
        x, dx, overall_forces = ctx.saved_tensors

        # Compute gradients of loss with respect to force
        ctx.tape.backward()

        dL_df_wp = ctx.tape.gradients[ctx.wp_vertice_forces]
        dL_df = wp.to_torch(dL_df_wp)

        dL_dv_before_collision = wp.to_torch(ctx.tape.gradients[ctx.wp_v_before_collision])
        dL_dv_before_ground = wp.to_torch(ctx.tape.gradients[ctx.wp_v_before_ground])

        # Compute gradients of force with respect to inputs
        df_dx = torch.autograd.grad(
            overall_forces, x, grad_outputs=dL_df, retain_graph=True, create_graph=True
        )[0]
        df_dv = torch.autograd.grad(
            overall_forces, dx, grad_outputs=dL_df, retain_graph=True, create_graph=True
        )[0]

        dL_dx = dL_df * df_dx
        dL_dv = dL_df * df_dv

        return (
            dL_dx,                    # x
            dL_dv,                    # v
            dL_dv_before_collision,   # v_before_collision
            dL_dv_before_ground,      # v_before_ground
            dL_df,                    # vertice_forces
            None,                     # num_object_points
            None,                     # drag_damping
            None,                     # wp_collide_elas
            None,                     # wp_collide_fric
            None,                     # reverse_factor
            None,                     # dt
            None,                     # tape
            None,                     # num_substeps
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
            logger.info(f"[SIMULATION]: Initialize the Spring-Mass Hybrid System")
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
            self.gt_object_points = gt_object_points+1
            if cfg.data_type == "real":
                self.gt_object_visibilities = gt_object_visibilities.int()
                self.gt_object_motions_valid = gt_object_motions_valid.int()

            self.num_surface_points = num_surface_points
            self.num_original_points = num_original_points
            if num_original_points is None:
                self.num_original_points = self.num_object_points

            # # Do some initialization to initialize the warp cuda graph
            self.wp_springs = wp.from_torch(
                init_springs, dtype=wp.vec2i, requires_grad=False
            )
            self.wp_rest_lengths = wp.from_torch(
                init_rest_lengths, dtype=wp.float32, requires_grad=False
            )
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

                self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
                self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
                self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

            # Initialize the warp parameters
            self.wp_states = []
            for i in range(self.num_substeps + 1):
                state = State(self.wp_init_velocities, self.num_control_points)
                self.wp_states.append(state)
            if cfg.data_type == "real":
                self.distance_matrix = wp.zeros(
                    (self.num_original_points, self.num_surface_points), requires_grad=False
                )
                self.neigh_indices = wp.zeros(
                    (self.num_original_points), dtype=wp.int32, requires_grad=False
                )

            # Parameter to be optimized
            self.wp_spring_Y = wp.from_torch(
                torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
                * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
                requires_grad=True,
            )
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

            #Create the CUDA graph to acclerate
            if cfg.use_graph:
                if cfg.data_type == "real":
                    if not disable_backward:
                        with wp.ScopedCapture() as capture:
                            self.loss = self.step()
                            self.loss.backward()
                    else:
                        with wp.ScopedCapture() as capture:
                            self.loss = self.step()
                    self.graph = capture.graph
                elif cfg.data_type == "synthetic":
                    if not disable_backward:
                        # For synthetic data, we compute simple loss
                        with wp.ScopedCapture() as capture:
                            self.loss = self.step()
                            self.loss.backward()
                    else:
                        with wp.ScopedCapture() as capture:
                            self.loss = self.step()
                    self.graph = capture.graph
                else:
                    raise NotImplementedError

                with wp.ScopedCapture() as forward_capture:
                    self.step()
                self.forward_graph = forward_capture.graph
            else:
                self.tape = wp.Tape()

    def step(self):
        final_loss = 0
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()

            final_loss, next_x, next_y, vertice_forces, v_before_collision, v_before_ground = MassSpringIntegrator.apply(
                self.states[i].wp_x,
                self.states[i].wp_v,
                self.states[i].wp_v_before_collision,
                self.states[i].wp_v_before_ground,
                self.states[i].wp_vertice_forces,
                self.states[i].wp_v,
                self.object_collision_flag, 
                self.num_object_points, 
                self.wp_masses, 
                self.dt, 
                self.drag_damping, 
                self.reverse_factor, 
                self.wp_collide_elas, 
                self.wp_collide_fric, 
                self.collision_dist, 
                self.wp_collision_indices, 
                self.wp_collision_number,
                self.wp_collide_object_elas,
                self.wp_collide_object_fric,
                self.wp_masks,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.wp_current_object_motions_valid,
                self.num_valid_visibilities,
                self.num_valid_motions,
                self.prev_acc,
                self.acc_count,
                cfg,
                self.num_original_points,
                self.neigh_indices,
                self.num_surface_points
            )

            self.states[i+1].wp_x = next_x
            self.states[i+1].wp_v = next_y
            self.states[i].wp_v_before_collision = v_before_collision
            self.states[i].wp_vertice_forces = vertice_forces
            self.states[i].wp_v_before_ground = v_before_ground
        
        return final_loss