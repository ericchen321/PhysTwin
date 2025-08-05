import torch
import warp as wp

from kernels import (
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

def eval_springs_energy():
    pass

class MassSpringIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        wp_x,
        wp_v,
        wp_v_before_collision,
        wp_v_before_ground,
        wp_vertice_forces,
        wp_v_initial,
        object_collision_flag, 
        num_object_points, 
        wp_masses, 
        dt, 
        drag_damping, 
        reverse_factor, 
        wp_collide_elas, 
        wp_collide_fric, 
        collision_dist, 
        wp_collision_indices, 
        wp_collision_number,
        wp_collide_object_elas,
        wp_collide_object_fric,
        wp_masks,
        wp_current_object_points,
        wp_current_object_visibilities,
        wp_current_object_motions_valid,
        num_valid_visibilities,
        num_valid_motions,
        prev_acc,
        acc_count,
        cfg,
        num_original_points,
        neigh_indices,
        num_surface_points
    ):
        # evaluate the spring energies using pytorch
        elastic_energy, damping_energy = eval_springs_energy()

        x = wp.to_torch(wp_x)
        dx = wp.to_torch(wp_v)

        elastic_force = -torch.autograd.grad(elastic_energy, x, create_graph=True)
        damping_force = -torch.autograd.grad(damping_energy, x, create_graph=True)

        overall_force = elastic_force + damping_force

        wp_vertice_forces = wp.from_torch(overall_force, dtype=wp.vec3, requires_grad=True)

        # simulate the remainder of the forward pass using warp
        ctx.tape = wp.Tape()
        with ctx.tape:
            if object_collision_flag:
                output_v = wp_v_before_collision
            else:
                output_v = wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=num_object_points,
                inputs=[
                    wp_v,
                    wp_vertice_forces,
                    wp_masses,
                    dt,
                    drag_damping,
                    reverse_factor,
                ],
                outputs=[output_v],
            )

            if object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=num_object_points,
                    inputs=[
                        wp_x,
                        wp_v_before_collision,
                        wp_masses,
                        wp_masks,
                        wp_collide_object_elas,
                        wp_collide_object_fric,
                        collision_dist,
                        wp_collision_indices,
                        wp_collision_number,
                    ],
                    outputs=[wp_v_before_ground],
                )

            # Update the x and v
            next_x = wp.zeros_like(wp_x, requires_grad=True)
            next_y = wp.zeros_like(wp_v, requires_grad=True)
            wp.launch(
                kernel=integrate_ground_collision,
                dim=num_object_points,
                inputs=[
                    wp_x,
                    wp_v_before_ground,
                    wp_collide_elas,
                    wp_collide_fric,
                    dt,
                    reverse_factor,
                ],
                outputs=[next_x, next_y],
            )

            # Calculate loss

            chamfer_loss, track_loss, acc_loss, loss = 0
            distance_matrix = wp.zeros(
                (num_original_points, num_surface_points), requires_grad=False
            )

            # Compute the chamfer loss
            # Precompute the distances matrix for the chamfer loss
            # TODO: does states[-1].x refer to the current state or a future state?
            wp.launch(
                compute_distances,
                dim=(num_original_points, num_surface_points),
                inputs=[
                    wp_x,
                    wp_current_object_points,
                    wp_current_object_visibilities,
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
                    wp_x,
                    wp_current_object_points,
                    wp_current_object_visibilities,
                    num_valid_visibilities,
                    neigh_indices,
                    cfg.chamfer_weight,
                ],
                outputs=[chamfer_loss],
            )

            # Compute the tracking loss
            wp.launch(
                compute_track_loss,
                dim=num_original_points,
                inputs=[
                    wp_x,
                    wp_current_object_points,
                    wp_current_object_motions_valid,
                    num_valid_motions,
                    cfg.track_weight,
                ],
                outputs=[track_loss],
            )

            wp.launch(
                compute_acc_loss,
                dim=num_object_points,
                inputs=[
                    wp_v_initial,
                    wp_v,
                    prev_acc,
                    num_object_points,
                    acc_count,
                    cfg.acc_weight,
                ],
                outputs=[acc_loss],
            )

            wp.launch(
                compute_final_loss,
                dim=1,
                inputs=[chamfer_loss, track_loss, acc_loss],
                outputs=[loss],
            )

        # TODO: honestly no idea what gradients are needed here.
        ctx.save_for_backward(
            loss,
            overall_force,
            x,
            dx
        )
        
        ctx.wp_vertice_forces = wp_vertice_forces
        ctx.wp_v_before_collision = wp_v_before_collision
        ctx.wp_v_before_ground = wp_v_before_ground

        return next_x, next_y, loss # for now we won't return the other losses.

    @staticmethod
    def backward(ctx, dL_dL, dL_dnext_x, dL_dnext_y):
        # Get saved tensors
        loss, overall_force, x, dx = ctx.saved_tensors[0] 

        # Compute gradients of loss with respect to force (dL_df)
        ctx.tape.backward(ctx.loss)

        dL_df_wp = ctx.tape.gradients[ctx.wp_vertice_forces]
        dL_df = wp.to_torch(dL_df_wp)

        # Compute gradients of loss with respect to inputs after force
        dL_dv_before_collision = wp.to_torch(ctx.tape.gradients[ctx.wp_v_before_collision])
        dL_dv_before_ground = wp.to_torch(ctx.tape.gradients[ctx.wp_v_before_ground])

        # Compute gradients of force with respect to inputs (df_dx, df_dv)
        df_dx = torch.autograd.grad(
            overall_force, x, grad_outputs=dL_df, retain_graph=True, create_graph=True
        )[0]
        df_dv = torch.autograd.grad(
            overall_force, dx, grad_outputs=dL_df, retain_graph=True, create_graph=True
        )[0]

        # Chain rule to get gradients of loss with respect to inputs (dL_dx, dL_dv)
        dL_dx = dL_df * df_dx
        dL_dv = dL_df * df_dv

        return (
            dL_dx,
            dL_dv,
            dL_dv_before_collision,
            dL_dv_before_ground,
            dL_df,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

