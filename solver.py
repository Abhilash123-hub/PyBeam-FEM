import numpy as np
from preprocessor import PreProcessor

class BeamFEM:
    def __init__(self, data):
        self.nodes = data['nodes']
        self.elements = data['elements']
        self.bcs = data['boundary_conditions']
        self.loads = data['loads']
        
        # 2 Degrees of Freedom (DOFs) per node: Vertical displacement (v) and Rotation (theta)
        self.num_nodes = len(self.nodes)
        self.total_dof = 2 * self.num_nodes
        
        # Initialize [K], {f}, and {u}
        self.K_global = np.zeros((self.total_dof, self.total_dof))
        self.F_global = np.zeros(self.total_dof)
        self.U_global = np.zeros(self.total_dof)

    def build_global_matrix(self):
        print("\nBuilding Global Stiffness Matrix...")
        for el in self.elements:
            n1 = el['node_start']
            n2 = el['node_end']
            
            x1 = self.nodes[n1]['x']
            x2 = self.nodes[n2]['x']
            L = abs(x2 - x1)
            
            E = el['E'] 
            I = el['I'] 
            
            k_e = (E * I / L**3) * np.array([
                [ 12,      6*L,     -12,      6*L   ],
                [ 6*L,     4*L**2,  -6*L,     2*L**2],
                [-12,     -6*L,      12,     -6*L   ],
                [ 6*L,     2*L**2,  -6*L,     4*L**2]
            ])
            
            dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            for i in range(4):
                for j in range(4):
                    self.K_global[dof_map[i], dof_map[j]] += k_e[i, j]

    def apply_loads_and_bcs(self):
        print("Applying External Loads...")
        for load in self.loads:
            node = load['node_id']
            self.F_global[2*node] += load['force_y']
            self.F_global[2*node + 1] += load['moment']

        print("Enforcing Boundary Conditions...")
        penalty = 1e15 # A massive number to lock the node in place
        for bc in self.bcs:
            node = bc['node_id']
            if bc['type'] == 'fixed':
                self.K_global[2*node, 2*node] += penalty      # Lock vertical (v)
                self.K_global[2*node+1, 2*node+1] += penalty  # Lock rotation (theta)
            elif bc['type'] == 'roller' or bc['type'] == 'pinned':
                self.K_global[2*node, 2*node] += penalty      # Lock vertical (v) only

    def solve(self):
        print("Solving the linear system [K]{u} = {f}...")
        # Solves the matrix equation computationally
        self.U_global = np.linalg.solve(self.K_global, self.F_global)
        
        print("\n--- NODAL DISPLACEMENTS ---")
        for i in range(self.num_nodes):
            v = self.U_global[2*i]
            theta = self.U_global[2*i+1]
            print(f"Node {i}: Deflection = {v*1000:.4f} mm | Rotation = {theta:.6f} rad")
        return self.U_global

# Test the solver
if __name__ == "__main__":
    parser = PreProcessor('beam_config.json')
    system_data = parser.load_data()
    
    if system_data:
        solver = BeamFEM(system_data)
        solver.build_global_matrix()
        solver.apply_loads_and_bcs()
        solver.solve()