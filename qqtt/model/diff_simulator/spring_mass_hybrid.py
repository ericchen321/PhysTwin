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
    spring_l0s: torch.Tensor,
    num_springs: float,
    spring_k: float,
    dashpot_damping: float
):
    t_spring_k = torch.Tensor([spring_k], device="cuda")
    t_spring_b = torch.Tensor([dashpot_damping], device="cuda")

    num_samples, num_particles, _ = x.shape
    epsi_ks = 1e-6
    epsi_bs = 1e-6

    # expand spring k and b, and apply positivity func
    spring_ks = torch.broadcast_to(
        t_spring_k.view(1, 1, 1),
        (1, num_springs, 1))
    spring_ks_pos = torch.nn.ReLU(spring_ks) + epsi_ks
    spring_bs = torch.broadcast_to(
        t_spring_b.view(1, 1, 1),
        (1, num_springs, 1))
    spring_bs_pos = torch.nn.ReLU(spring_bs) + epsi_bs

    # extract end point positions of each spring
    spring_pos = x[:, springs].view(
        num_samples, num_springs, 2, 3)
    
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
    
    spring_bs_pos = torch.nn.ReLU(
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