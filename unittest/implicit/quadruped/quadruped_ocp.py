import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer
import json

import sys
sys.path.append("unittest/implicit/quadruped")
from helper import *


def SolveOcp():
    opti = casadi.Opti()

    dt = 0.02
    N = 100

    quad = QuadrupedDynamics(timestep=dt)
    rhs = quad.acc_func

    qq = opti.variable(quad.model.nq, N+1)
    vv = opti.variable(quad.model.nv, N+1)
    uu = opti.variable(12, N)

    q0 = quad.model.referenceConfigurations["standing"]
    q0 = pin.normalize(quad.model, q0)
    q0[2] += 0.05

    v0 = np.zeros(quad.model.nv)
    v0[0] = 1.5
    v0[1] = 2.0

    opti.subject_to(qq[:,0] == q0)
    opti.subject_to(vv[:,0] == v0)

    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]

        x = casadi.vertcat(qk, vk)
        x_next = quad.forward_casadi(x, uk)
        q_next = x_next[:quad.model.nq]
        v_next = x_next[quad.model.nq:]

        opti.subject_to(qq[:,k+1] == q_next)
        opti.subject_to(vv[:,k+1] == v_next)

        # torque limits
        opti.subject_to(opti.bounded(-50, uk, 50))

    control_penalty = casadi.sumsqr(uu)
    terminal_standing_still_penalty = casadi.sumsqr(vv[:,N])
    n = 0
    reference_joint_stance = quad.model.referenceConfigurations["standing"][n:]
    reference_body_position = reference_joint_stance[0:3]
    reference_body_orientation = reference_joint_stance[3:7]
    reference_leg_joints = reference_joint_stance[7:]
    reference_body_position_penalty = casadi.sumsqr(qq[0:3,N] - reference_body_position.reshape(-1,1))
    reference_body_orientation_penalty = casadi.sumsqr(qq[3:7,N] - reference_body_orientation.reshape(-1,1))
    reference_leg_joints_penalty = casadi.sumsqr(qq[7:,N] - reference_leg_joints.reshape(-1,1))
    terminal_reference_penalty = casadi.sumsqr(qq[3:,N] - reference_joint_stance[3:].reshape(-1,1))

    movement_penalty = casadi.sumsqr(vv)
    body_movement_penalty = casadi.sumsqr(vv[:3,:])

    opti.minimize(0*control_penalty + 
                  1e3*terminal_standing_still_penalty + 
                  1e1*reference_body_position_penalty +
                  1e3*reference_body_orientation_penalty + 
                  0*reference_leg_joints_penalty + 
                  1e0*movement_penalty + 
                  1e1*body_movement_penalty +
                  1e3*terminal_reference_penalty)

    opti.set_initial(qq, np.tile(q0.reshape(-1,1), (1,N+1)))
    for k in range(N):
        opti.set_initial(uu[:,k], 
                         [-4.27557097,   5.91086644,  17.19719031,  -3.27040327,  
                          -4.81348097, -12.73962232,   3.67687864,   4.22256038,  
                          10.28272374,   5.82689115,  -3.42997252, -14.4580922])

    p_opts = {"expand": True}
    s_opts = {}
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()

    q_opt = sol.value(qq)
    v_opt = sol.value(vv)
    u_opt = sol.value(uu)
    t_opt = np.arange(N+1)*dt
    return {"t": t_opt, "q": q_opt, "v": v_opt, "u": u_opt, "dt": dt, "N": N}, \
           {"model": quad.model, "visual_model": quad.visual_model, 
            "collision_model": quad.collision_model}

def visualize_ocp_result(result, models):
    N = result["N"]
    q_opt = result["q"]
    v_opt = result["v"]
    states_ = []
    for k in range(N+1):
        xk = np.concatenate((q_opt[:,k], v_opt[:,k]))
        states_.append(xk)

        if k == 0:
            # copy the first state to have a longer video
            time_to_add = 2.0
            steps_to_add = int(np.ceil(time_to_add / result["dt"]))
            for _ in range(steps_to_add):
                states_.append(states_[0])
    
    time = result["t"]
    controls = result["u"]

    ##########################
    ### VISUALIZE CONTROLS ###
    ##########################
    import matplotlib.pyplot as plt
    
    hip_abduction_color = "tab:blue"
    hip_flexion_color = "tab:orange"
    knee_flexion_color = "tab:green"
    legs = ["LF", "RF", "LH", "RH"]

    print("Control dims:")
    print(controls.shape)

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        plt.plot(time[:-1], controls[i*3+0,:], label="HAA", color=hip_abduction_color)
        plt.plot(time[:-1], controls[i*3+1,:], label="HFE", color=hip_flexion_color)
        plt.plot(time[:-1], controls[i*3+2,:], label="KFE", color=knee_flexion_color)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.xlim([time[0], time[-1]])

    # create space below for a single legend
    plt.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_controls.png", dpi=300)
    # plt.show()

    print(f"controls in last time-step: {controls[:,-1]}")

    ##############################
    ### VISUALIZE FOOT HEIGHTS ###
    ##############################
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        foot_id = models["model"].getFrameId(foot_names[i])
        heights = []
        for k in range(N+1):
            qk = np.array(q_opt[:,k])
            data = models["model"].createData()
            pin.forwardKinematics(models["model"], data, qk)
            pin.updateFramePlacements(models["model"], data)
            p = data.oMf[foot_id].translation
            heights.append(p[2])
        plt.plot(time, heights)
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.xlim([time[0], time[-1]])
    
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_foot_heights.png", dpi=300)
    # plt.show()

    ###################################
    ### VISUALIZE ROBOT BODY MOTION ###
    ###################################
    plt.figure()
    for i in range(3):
        plt.subplot(3,1,i+1)
        if i == 0:
            plt.title("Body Position")
            plt.ylabel("X [m]")
        elif i == 1:
            plt.ylabel("Y [m]")
        else:
            plt.ylabel("Z [m]")
            plt.xlabel("Time [s]")
        plt.plot(time, q_opt[i,:])
        plt.xlim([time[0], time[-1]])
    
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_body_position.png", dpi=300)
    # plt.show()
    
    #################################
    ### VISUALIZE REACTION FORCES ###
    #################################
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        fx = []
        fy = []
        fz = []
        for k in range(N+1):
            qk = np.array(q_opt[:,k])
            vk = np.array(v_opt[:,k])
            data = models["model"].createData()
            pin.forwardKinematics(models["model"], data, qk, vk)
            pin.updateFramePlacements(models["model"], data)
            foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
            foot_id = models["model"].getFrameId(foot_names[i])
            p = data.oMf[foot_id].translation
            v_frame = pin.getFrameVelocity(models["model"], data, foot_id, pin.LOCAL_WORLD_ALIGNED)
            v_foot = v_frame.linear
            f_contact = ground_reaction_force(p, v_foot)
            fx.append(f_contact[0])
            fy.append(f_contact[1])
            fz.append(f_contact[2])
        plt.plot(time, fx, label="fx")
        plt.plot(time, fy, label="fy")
        plt.plot(time, fz, label="fz")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.xlim([time[0], time[-1]])
    
    # create space below for a single legend
    plt.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_reaction_forces.png", dpi=300)
    # plt.show()
    plt.close()

    ##############################
    ### VISUALIZE ROBOT MOTION ###
    ##############################
    states_ = np.stack(states_).T
    viz = MeshcatVisualizer(
        model=models["model"],
        collision_model=models["collision_model"],
        visual_model=models["visual_model"],
    )
    qs_ = states_[: models["model"].nq, :].T
    viz.initViewer()
    viz.loadViewerModel("pinocchio")
    import time
    time.sleep(5.0)
    viz.play(q_trajectory=qs_, dt=results["dt"])

    time.sleep(N*results["dt"] + 5.0)
           

if __name__ == "__main__":
    quad = QuadrupedDynamics(timestep=0.02)
    quad.test_dynamics()

    folder_name = "unittest/implicit/quadruped/json_files/"

    # load settings file
    with open(folder_name + "quadruped_ocp_settings.json", "r") as f:
        settings = json.load(f)

    if not settings["LOAD_OCP_SOLUTION"]:
        results, models = SolveOcp()

        if settings["STORE_OCP_SOLUTION"]:
            # save results to a json file
            with open(folder_name + "quadruped_ocp_solution.json", "w") as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f)

    else:
        # load results from a json file
        with open(folder_name + "quadruped_ocp_solution.json", "r") as f:
            results = json.load(f)
        results["q"] = np.array(results["q"])
        results["v"] = np.array(results["v"])
        results["u"] = np.array(results["u"])
        results["t"] = np.array(results["t"])
        from pinocchio.robot_wrapper import RobotWrapper
        robot = erd.load("anymal")
        models = {}
        models["model"] = robot.model
        models["collision_model"] = robot.collision_model
        models["visual_model"] = robot.visual_model

    visualize_ocp_result(results, models)
