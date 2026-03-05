import numpy as np
import matplotlib.pyplot as plt
from preprocessor import PreProcessor
from solver import BeamFEM

class PostProcessor:
    def __init__(self, nodes, elements, U_global):
        self.nodes = nodes
        self.elements = elements
        self.U_global = U_global
        self.scale_factor = 100 # We exaggerate the tiny deflections so we can actually see them

    def plot_deformation(self):
        print(f"\nPlotting deformation (scaled by {self.scale_factor}x)...")
        plt.figure(figsize=(10, 4))
        
        # Plot original nodes and elements
        for el in self.elements:
            n1 = el['node_start']
            n2 = el['node_end']
            x1 = self.nodes[n1]['x']
            x2 = self.nodes[n2]['x']
            
            # Draw the original unbent beam (dashed black line)
            plt.plot([x1, x2], [0, 0], 'k--', linewidth=2, label='Original' if el['id']==0 else "")
            
            # --- MATH: Hermite Shape Functions to draw a smooth curve ---
            L = abs(x2 - x1)
            x_vals = np.linspace(0, L, 50) # 50 points along the element
            xi = x_vals / L
            
            # Extract the calculated displacements and rotations for this specific element
            v1 = self.U_global[2*n1]
            theta1 = self.U_global[2*n1 + 1]
            v2 = self.U_global[2*n2]
            theta2 = self.U_global[2*n2 + 1]
            
            # The Hermite equations
            N1 = 1 - 3*xi**2 + 2*xi**3
            N2 = L * (xi - 2*xi**2 + xi**3)
            N3 = 3*xi**2 - 2*xi**3
            N4 = L * (-xi**2 + xi**3)
            
            # Interpolate the vertical displacement for a smooth plot
            v_interpolated = N1*v1 + N2*theta1 + N3*v2 + N4*theta2
            
            # Convert to global coordinates and apply the visual scale factor
            x_global = x1 + x_vals
            y_global = v_interpolated * self.scale_factor
            
            # Draw the deformed beam (solid blue line)
            plt.plot(x_global, y_global, 'b-', linewidth=3, label='Deformed' if el['id']==0 else "")
        
        # Plot the original nodes (black dots)
        orig_x = [node['x'] for node in self.nodes]
        plt.plot(orig_x, [0]*len(orig_x), 'ko', markersize=8)
        
        # Plot the displaced nodes (red dots)
        def_y = [self.U_global[2*i] * self.scale_factor for i in range(len(self.nodes))]
        plt.plot(orig_x, def_y, 'ro', markersize=8)
        
        # Add labels and formatting
        plt.title(f'Beam Deformation (Exaggerated {self.scale_factor}x)')
        plt.xlabel('Length along the beam (m)')
        plt.ylabel('Deflection')
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.show()

# Run the full pipeline
if __name__ == "__main__":
    parser = PreProcessor('beam_config.json')
    system_data = parser.load_data()
    
    if system_data:
        # 1. Solve the math
        solver = BeamFEM(system_data)
        solver.build_global_matrix()
        solver.apply_loads_and_bcs()
        U = solver.solve()
        
        # 2. Visualize it
        visualizer = PostProcessor(system_data['nodes'], system_data['elements'], U)
        visualizer.plot_deformation()