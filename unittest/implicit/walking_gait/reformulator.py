from casadi import *
from collocation_scheme import collocation_scheme
import matplotlib.pyplot as plt

class Reformulator:
    def __init__(self, n_coll, n_coords, n_act_mesh, dt, xk, uk, uk_orig, qcoll, qdotcoll, qddotcoll):
        [tau, C, D, B] = collocation_scheme(n_coll, 'radau')
        self.C = C[:,1:]
        print(f"self.C: {self.C}")
        assert n_coll == 3
        assert n_coords == 9
        assert n_act_mesh == 18
        self.n_coll = n_coll
        self.n_coords = n_coords
        self.dt = dt
        self.xk = xk
        self.uk = uk
        self.uk_orig = uk_orig
        self.n_act_mesh = n_act_mesh
        
        self.keep_orginal_constraint_order = True
        self.substitute_q = False
        self.substitute_qdot = False        
        self.eliminate_in_dynamics_equations = True
        
        self.q_mesh = self.xk[:self.n_coords]
        self.qdot_mesh = self.xk[self.n_coords:2*self.n_coords]
        
        self.q_coll = qcoll
        self.qdot_coll = qdotcoll
        self.qddot_coll = qddotcoll
        
        self.scale_qdots = 10
        self.scale_qddots = 100
        
        self.reconstruct_err_coll()
        
    def show_gk_jacobian_structure_for_debugging(self, f_gk):
        fig = plt.figure()
        # show jacobian
        plt.spy(jacobian(f_gk(self.xk, self.uk), vertcat(self.xk, self.uk)).sparsity())
        
        alpha = 0.2
        zorder = 1
        
        # for illustrations, highlight different block structures
        n_rows = f_gk(self.xk, self.uk).shape[0]
        col_ptr = 0
        # show states
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords, n_rows, color='red', lw=0, alpha=alpha, zorder=zorder, label='[q_mesh, qdot_mesh]'))
        col_ptr += 2*self.n_coords
        
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords - 1, n_rows, color='darkred', lw=0, alpha=alpha, zorder=zorder, label='[q0_mesh, q0dot_mesh]'))
        col_ptr += 2*self.n_coords - 1
        
        # show controls
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), self.n_act_mesh, n_rows, color='blue', lw=0, alpha=alpha, zorder=zorder, label='act_mesh'))
        col_ptr += self.n_act_mesh
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 3*self.n_coords, n_rows, color='blue', lw=0, alpha=alpha, zorder=zorder, label='[q_coll, qdot_coll, qddot_coll]'))
        col_ptr += 3*self.n_coords
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 3*self.n_coords, n_rows, color='darkblue', lw=0, alpha=alpha, zorder=zorder, label='[q_coll, qdot_coll, qddot_coll]'))
        col_ptr += 3*self.n_coords
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), self.n_coords, n_rows, color='blue', lw=0, alpha=alpha, zorder=zorder, label='[qddot_coll]'))
        col_ptr += self.n_coords
        
        # show zk variables
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords, n_rows, color='green', lw=0, alpha=alpha, zorder=zorder, label='[qk, qdotk]'))
        col_ptr += 2*self.n_coords
        
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords - 1, n_rows, color='darkgreen', lw=0, alpha=alpha, zorder=zorder, label='[q0_mesh, q0dot_mesh]'))
        col_ptr += 2*self.n_coords - 1
        
        # plt.legend()
        plt.savefig(f'jacobian_structure_{f_gk.name()}.png', dpi=300)
        plt.close()
        

    def reconstruct_err_coll(self):
        qc1 = self.q_coll[:self.n_coords]
        qc2 = self.q_coll[self.n_coords:2*self.n_coords]
        qc3 = self.q_coll[2*self.n_coords:3*self.n_coords]
        qc1dot = self.qdot_coll[:self.n_coords]
        qc2dot = self.qdot_coll[self.n_coords:2*self.n_coords]
        qc3dot = self.qdot_coll[2*self.n_coords:3*self.n_coords]
        qc1ddot = self.qddot_coll[:self.n_coords]
        qc2ddot = self.qddot_coll[self.n_coords:2*self.n_coords]
        qc3ddot = self.qddot_coll[2*self.n_coords:3*self.n_coords]
        
        scale_qdots = self.scale_qdots
        scale_qddots =  self.scale_qddots
        if self.substitute_q:
            qc3_subs = -1.0/self.C[3,0] * (self.C[0,0]*self.q_mesh + self.C[1,0]*qc1 + self.C[2,0]*qc2 - scale_qdots*self.dt*qc1dot)
        else:
            qc3_subs = qc3
        
        if self.substitute_qdot:
            qc3dot_subs = -1.0/self.C[3,0] * (self.C[0,0]*self.qdot_mesh + self.C[1,0]*qc1dot + self.C[2,0]*qc2dot - scale_qddots*self.dt*qc1ddot/scale_qdots)
        else:
            qc3dot_subs = qc3dot

        err_coll_q_reconstructed = vertcat(
            self.C[0,0]*self.q_mesh + self.C[1,0]*qc1 + self.C[2,0]*qc2 + self.C[3,0]*qc3 - scale_qdots*self.dt*qc1dot,
            self.C[0,1]*self.q_mesh + self.C[1,1]*qc1 + self.C[2,1]*qc2 + self.C[3,1]*qc3_subs - scale_qdots*self.dt*qc2dot,
            self.C[0,2]*self.q_mesh + self.C[1,2]*qc1 + self.C[2,2]*qc2 + self.C[3,2]*qc3_subs - scale_qdots*self.dt*qc3dot_subs
        )
        
        err_coll_qdot_reconstructed = vertcat(
            scale_qdots*self.C[0,0]*self.qdot_mesh + 
                scale_qdots*self.C[1,0]*qc1dot + 
                scale_qdots*self.C[2,0]*qc2dot + 
                scale_qdots*self.C[3,0]*qc3dot - 
                scale_qddots*self.dt*qc1ddot, 
            scale_qdots*self.C[0,1]*self.qdot_mesh + 
                scale_qdots*self.C[1,1]*qc1dot + 
                scale_qdots*self.C[2,1]*qc2dot + 
                scale_qdots*self.C[3,1]*qc3dot_subs - 
                scale_qddots*self.dt*qc2ddot,
            scale_qdots*self.C[0,2]*self.qdot_mesh + 
                scale_qdots*self.C[1,2]*qc1dot + 
                scale_qdots*self.C[2,2]*qc2dot + 
                scale_qdots*self.C[3,2]*qc3dot_subs - 
                scale_qddots*self.dt*qc3ddot
        )
        
        self.err_coll = Function('err_coll', [self.xk, self.uk], [vertcat(err_coll_q_reconstructed, err_coll_qdot_reconstructed/self.scale_qdots)])
        
        uk_substituted = vertcat(
            self.uk[:self.n_act_mesh],
            qc1,
            qc1dot,
            qc1ddot,
            qc2,
            qc2dot,
            qc2ddot,
            qc3ddot,
            qc3_subs,
            qc3dot_subs,
            self.uk[self.n_act_mesh + 3*self.n_coll*self.n_coords:] # the rest of the controls that are not collocation variables
        )
        
        self.uk_subs = Function('uk_subs', [self.xk, self.uk], [uk_substituted])
                
    def get_f_gk_reformulated(self, f_gk):
        if self.eliminate_in_dynamics_equations:
            err_sysdyn_rewritten = f_gk(self.xk, self.uk)[self.n_coll*self.n_coords*2:self.n_coll*self.n_coords*3]
        else:
            err_sysdyn_rewritten = f_gk(self.xk, self.uk_subs(self.xk, self.uk))[self.n_coll*self.n_coords*2:self.n_coll*self.n_coords*3]
        err_coll = self.err_coll(self.xk, self.uk)
        subs_q_coll_eqs = err_coll[:self.n_coords]
        other_q_coll_eqs = err_coll[self.n_coords:self.n_coll*self.n_coords]
        subs_qdot_coll_eqs = err_coll[self.n_coll*self.n_coords:self.n_coll*self.n_coords + self.n_coords]
        other_qdot_coll_eqs = err_coll[self.n_coll*self.n_coords + self.n_coords:]
                
        if self.keep_orginal_constraint_order:
            err_coll_arranged = vertcat(
                subs_q_coll_eqs,            # original order
                other_q_coll_eqs,
                subs_qdot_coll_eqs,
                other_qdot_coll_eqs,
            )
        else:
            err_coll_arranged = vertcat(
                other_q_coll_eqs,           # desired order
                other_qdot_coll_eqs,
                subs_q_coll_eqs,
                subs_qdot_coll_eqs
            )
        
        
        periodic_constraints = f_gk(self.xk, self.uk)[self.n_coll*self.n_coords*3:]
        if self.keep_orginal_constraint_order:
            self.f_gk_reformulated = Function('f_gk_reformulated', 
                                                [self.xk, self.uk], 
                                                [vertcat(
                                                    err_sysdyn_rewritten,     # desired order
                                                    err_coll_arranged,
                                                    periodic_constraints
                                                    )])
        else:
            self.f_gk_reformulated = Function('f_gk_reformulated', 
                                                [self.xk, self.uk], 
                                                [vertcat(
                                                    err_coll_arranged,        # desired order
                                                    err_sysdyn_rewritten, 
                                                    periodic_constraints
                                                    )])
        return self.f_gk_reformulated
        