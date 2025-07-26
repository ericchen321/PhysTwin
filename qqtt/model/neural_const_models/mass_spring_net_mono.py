# Author: Ganidhu, Guanxiong
# Warp implementation of MassSpringNetMono

import warp as wp
import numpy as np
from typing import Dict, Tuple


@wp.kernel
def compute_spring_energies(
    x: wp.array(dtype=wp.vec3),           # particle positions
    dx: wp.array(dtype=wp.vec3),          # particle velocities  
    springs: wp.array(dtype=wp.vec2i),    # spring connectivity (particle indices)
    spring_l0s: wp.array(dtype=float),    # rest lengths
    spring_k: float,                      # spring stiffness (scalar)
    spring_b: float,                      # damping coefficient (scalar)
    energy_elastic: wp.array(dtype=float), # output elastic energies
    energy_damping: wp.array(dtype=float)  # output damping energies
):
    """
    Warp kernel to compute elastic and damping energies for all springs in parallel.
    """
    spring_idx = wp.tid()
    
    # Get spring endpoints
    p0_idx = springs[spring_idx][0]
    p1_idx = springs[spring_idx][1]
    
    # Get positions and velocities
    pos0 = x[p0_idx]
    pos1 = x[p1_idx]
    vel0 = dx[p0_idx]
    vel1 = dx[p1_idx]
    
    # Compute spring vector and length
    d = pos1 - pos0
    spring_length = wp.length(d)
    
    # Get rest length
    l0 = spring_l0s[spring_idx]
    
    # Compute elastic energy: 0.5 * k * (l - l0)^2
    l_minus_l0 = spring_length - l0
    elastic_energy = 0.5 * spring_k * l_minus_l0 * l_minus_l0
    
    # Compute strain
    strain = l_minus_l0 / (l0 + 1e-6)
    
    # Compute relative velocity
    vrel = vel1 - vel0
    
    # Compute unit direction vector
    d_unit = wp.normalize(d + wp.vec3(1e-6, 1e-6, 1e-6))  # add small epsilon to avoid division by zero
    
    # Project relative velocity onto spring direction
    vrel_proj_speed = wp.dot(vrel, d_unit)
    vrel_proj = vrel_proj_speed * d_unit
    
    # Compute strain rate
    # dstrain_dd = d / (l0 * spring_length + 1e-6)
    dstrain_dd_mag = 1.0 / (l0 + 1e-6)
    strain_rate = dstrain_dd_mag * vrel_proj_speed
    
    # Compute damping energy: 0.5 * b * (strain_rate)^2
    damping_energy = 0.5 * spring_b * strain_rate * strain_rate
    
    # Store results
    energy_elastic[spring_idx] = elastic_energy
    energy_damping[spring_idx] = damping_energy


class MassSpringNetMonoWarp:
    """
    Warp-based implementation of MassSpringNetMono.
    A variant of mass-spring net: 1) apply a single spring k, b to all springs;
    2) no topology learning; 3) no barrier force learning; 4) no spring limit
    learning. Works only for surrogates with one discretization.
    """
    
    def __init__(
        self,
        springs: np.ndarray,
        spring_l0s: np.ndarray,
        k_init: Dict = {
            "name": "normal",
            "a": 1.0,
            "b": 1.0
        },
        b_init: Dict = {
            "name": "normal", 
            "a": 1.0,
            "b": 1.0
        },
        device: str = "cuda"
    ) -> None:
        """
        Parameters:
            springs: Particle indices of each spring (num_springs, 2)
            spring_l0s: Rest lengths of the springs (num_springs,)
            k_init: Initialization method for spring stiffness
            b_init: Initialization method for spring damping coefficient
            device: Device to run the model ("cuda" or "cpu")
        """
        
        # Initialize Warp with specified device
        if device == "cuda":
            wp.init()
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Handle input shapes - remove singleton dimensions if present
        if len(springs.shape) == 3 and springs.shape[0] == 1:
            springs = springs[0]
        if len(spring_l0s.shape) == 3 and spring_l0s.shape[0] == 1:
            spring_l0s = spring_l0s[0].flatten()
        elif len(spring_l0s.shape) == 2:
            spring_l0s = spring_l0s.flatten()
            
        self.num_springs = springs.shape[0]
        
        # Create Warp arrays for spring topology and rest lengths
        self.springs = wp.array(springs.astype(np.int32), dtype=wp.vec2i, device=self.device)
        self.spring_l0s = wp.array(spring_l0s.astype(np.float32), dtype=float, device=self.device)
        
        # Initialize spring parameters
        self.spring_k = self._init_param(k_init)
        self.spring_b = self._init_param(b_init)
        
        # Ensure positive values
        self.epsi_ks = 1e-6
        self.epsi_bs = 1e-6
        
    def _init_param(self, init_config: Dict) -> float:
        """Initialize a parameter based on configuration."""
        init_name = init_config["name"]
        
        if init_name == "normal":
            a = init_config["a"]
            b = init_config["b"]
            return float(np.random.normal(a, b))
        elif init_name == "positive":
            return float(np.abs(1.0 + np.random.normal(0, 1)))
        else:
            raise ValueError(f"Invalid initialization strategy: {init_name}")
    
    def get_spring_ks(self) -> float:
        """Get the spring stiffness parameter."""
        return max(self.spring_k, self.epsi_ks)
    
    def get_spring_bs(self) -> float:
        """Get the spring damping parameter."""
        return max(self.spring_b, self.epsi_bs)
    
    def forward(
        self,
        x: wp.array,      # particle positions (num_particles,) of vec3
        dx: wp.array      # particle velocities (num_particles,) of vec3  
    ) -> Tuple[wp.array, wp.array]:
        """
        Forward pass.
        
        Parameters:
            x: particle positions as Warp array of vec3
            dx: particle velocities as Warp array of vec3
            
        Returns:
            Tuple of (energy_elastic, energy_damping) as Warp arrays
        """
        
        # Create output arrays
        energy_elastic = wp.zeros(self.num_springs, dtype=float, device=self.device)
        energy_damping = wp.zeros(self.num_springs, dtype=float, device=self.device)
        
        # Get positive parameters
        k_pos = self.get_spring_ks()
        b_pos = self.get_spring_bs()
        
        # Launch kernel
        wp.launch(
            kernel=compute_spring_energies,
            dim=self.num_springs,
            inputs=[
                x, dx, self.springs, self.spring_l0s,
                k_pos, b_pos
            ],
            outputs=[energy_elastic, energy_damping],
            device=self.device
        )
        
        return energy_elastic, energy_damping
    
    def forward_dict(
        self, 
        dict_in: Dict[str, np.ndarray]
    ) -> Dict[str, wp.array]:
        """
        Forward pass with dictionary input (for compatibility with original interface).
        
        Parameters:
            dict_in: dictionary containing
            - x: particle positions (num_samples, num_particles, 3) 
            - dx: particle velocities (num_samples, num_particles, 3)
            
        Returns:
            Dictionary containing
            - energy_elastic: elastic potential energy (num_springs,)
            - energy_damping: damping energy (num_springs,)
        """
        
        # Convert numpy arrays to Warp arrays
        # Assuming single sample (num_samples=1), take first sample
        x_np = dict_in["x"][0] if len(dict_in["x"].shape) == 3 else dict_in["x"]
        dx_np = dict_in["dx"][0] if len(dict_in["dx"].shape) == 3 else dict_in["dx"]
        
        # Create Warp arrays of vec3
        x_warp = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        dx_warp = wp.array(dx_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        
        # Compute energies
        energy_elastic, energy_damping = self.forward(x_warp, dx_warp)
        
        return {
            "energy_elastic": energy_elastic,
            "energy_damping": energy_damping
        }


# Example usage
if __name__ == "__main__":
    # Example setup
    wp.init()
    
    # Create a simple 3-particle, 2-spring system
    springs = np.array([[0, 1], [1, 2]], dtype=np.int32)
    spring_l0s = np.array([1.0, 1.0], dtype=np.float32)
    
    # Initialize the mass-spring network
    msn = MassSpringNetMonoWarp(
        springs=springs,
        spring_l0s=spring_l0s,
        k_init={"name": "normal", "a": 100.0, "b": 10.0},
        b_init={"name": "normal", "a": 1.0, "b": 0.1}
    )
    
    # Create particle positions and velocities
    num_particles = 3
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0], 
        [2.0, 0.1, 0.0]
    ], dtype=np.float32)
    
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0]
    ], dtype=np.float32)
    
    # Convert to Warp arrays
    x = wp.array(positions, dtype=wp.vec3, device="cuda")
    dx = wp.array(velocities, dtype=wp.vec3, device="cuda")
    
    # Compute energies
    energy_elastic, energy_damping = msn.forward(x, dx)
    
    # Print results
    print("Elastic energies:", energy_elastic.numpy())
    print("Damping energies:", energy_damping.numpy())