import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sdf import d2
from virtual_contact_2d import utils, BodyPlaneVirtualContacts, World, Body2D

# TODO:
# - [ ] Specify how the newton step happens (with or without contact recalculation)
# - [ ] Specify nr of newton iterations for sdf-sdf contact at the very beginning
# - [ ] Specify nr of newton iterations for sdf-sdf contact at every simulation step




####################################################################################################################################
####################################################################################################################################

def main():
    # ----------------------------------
    # Experiment parameters
    # ----------------------------------
    type_exp = 'lift_with_friction'
    N = 100
    ndof_u = 3
    os.makedirs(f'output/', exist_ok=True) # used by world to save images during forward pass
    os.makedirs(f'media/{type_exp}', exist_ok=True)

    # ----------------------------------
    # 1.a Define the SDF
    # ----------------------------------
    sdf_func = d2.rectangle(size=[1,2])
    sdf_func = d2.circle(radius=1.5)
    z0 = 1.5
    # sdf_func = d2.hexagon(r=1.5)
    # z0 = 1.5 / (2 *np.tan(np.pi/6))

    # ----------------------------------
    # Define the body
    # ----------------------------------
    body = Body2D(sdf_func, m=0.1, q0=torch.tensor([0., z0, 0.]), qdot0=torch.tensor([0., 0., 0.]))

    # ----------------------------------
    # Define the virtual contact points
    # ----------------------------------
    vc = BodyPlaneVirtualContacts(body, N_contacts=10, contact_init_iters=100, kn=1e3,
                                  damping=False, kd=5e1,
                                  smoothed_penalty=False, smoothed_alpha=1e-2,
                                  friction=True, kt=1e3, mu=0.1)

    # ----------------------------------
    # Define the virtual contact points
    # ----------------------------------
    sim = World(vc, dt=1./60., gravity=9.81, recompute_contacts_durint_optim=True, update_contacts_iters=10)

    # ----------------------------------
    # Initialize the input
    # ----------------------------------
    inputs = torch.zeros(N, ndof_u)
    # inputs[:,0] = 0.4                                # push to the right
    inputs[:,1] = torch.linspace(0, 15, N)* body.m     # lift up
    # inputs[:,0] = torch.linspace(0, 8, N)* circle.m  # rotation


    # --------------------------------------------------------------------
    # ------------------ Try out the non-smoothed case -------------------
    # --------------------------------------------------------------------
    vc.smoothed_penalty = False

    # ----------------------------------
    # a. Run the simulation forward
    # ----------------------------------
    qs, qdots = sim.forward(inputs)
    utils.create_gif('output', f'media/{type_exp}/output_normal.gif')

    # ----------------------------------
    # b. Run the simulation backward (compute gradients)
    # ----------------------------------
    dqdus, dqdqprevs, dqdqdotprevs = sim.backward_gradients(qs[1:], qs[:-1], qdots[:-1], inputs)

    # ----------------------------------
    # c. Plot the results
    # ----------------------------------
    fig, axs = plt.subplots(4, 3, figsize=(15, 12), dpi=80, sharex=True)
    ax_u = axs[0]
    ax_q = axs[1]
    ax_qdot = axs[2]
    ax_du = axs[3]

    label_qs = ['x', 'z', r'$\theta$']
    for i in range(ndof_u):
        ax_u[i].plot(inputs[:,i] / (body.m if i==1 else 1.), label='u_'+label_qs[i])
        ax_q[i].plot(qs[:, i], label=label_qs[i] )
        ax_qdot[i].plot(qdots[:, i], label= label_qs[i]+'_dot')
        ax_du[i].plot(np.stack([t[i,i] for t in dqdus]), label = 'd' + label_qs[i]+'_du')
    fig.savefig(f'media/{type_exp}/nonsmoothed_results2d.png')


    # --------------------------------------------------------------------
    # ------- Simulate with different alphas and visualize results -------
    # --------------------------------------------------------------------
    vc.smoothed_penalty = True

    for alpha in [0.001, 0.0008, 0.005]:
        vc.smoothed_alpha = alpha

        # a. Run the simulation forward
        qs_smoothed, qdots_smoothed = sim.forward(inputs)
        utils.create_gif('output', f'media/{type_exp}/output_smooth_{alpha}.gif')

        # b. Run the simulation backward (compute gradients)
        dqdus_smoothed, dqdqprevs_smoothed, dqdqdotprevs_smoothed = sim.backward_gradients(
            qs_smoothed[1:], qs_smoothed[:-1], qdots_smoothed[:-1], inputs)

        # c. Plot the results
        for i in range(ndof_u):
            ax_q[i].plot(qs_smoothed[:, i], label = label_qs[i] + fr', $\alpha={alpha}$')
            ax_qdot[i].plot(qdots_smoothed[:, i], label = label_qs[i] + fr'_dot, $\alpha={alpha}$')
            ax_du[i].plot(np.stack([t[i,i] for t in dqdus_smoothed]), label= 'd' + label_qs[i] + fr'_du, $\alpha={alpha}$')
            fig.savefig(f'media/{type_exp}/results2d.png')

    [ax.legend() for ax in axs.flatten()]
    plt.suptitle(f"{type_exp} with {vc.N_contacts} virtual contact points")
    fig.savefig(f'media/{type_exp}/results2d.pdf')
    fig.savefig(f'media/{type_exp}/results2d.png')


####################################################################################################################################

def visualize_contacts():
    """Visualize the contact points and the contact forces while moving the body"""
    sdf_func = d2.circle(radius=1.5)
    # sdf_func = d2.rectangle(size=[1,2])
    # sdf_func = d2.hexagon(r=1.5)
    z0 = 1.5 / (2 *np.tan(np.pi/6)) + 2

    circle = Body2D(sdf_func, m=0.1, q0=torch.tensor([0., z0, 0.]), qdot0=torch.tensor([0., 0., 0.]))

    inter = BodyPlaneVirtualContacts(circle)
    inter.update_contacts(circle.q, max_iter=100)

    N = 88
    path = 'image.png'
    for i in range(N+1):
        print(f"step {i}/{N}")
        # inter.body.q = 5*torch.tensor(
        #     [np.cos(2*i/N*np.pi),
        #      np.sin(2*i/N*np.pi),
        #      np.sin(i/N*np.pi),
        #      ])
        inter.body.q = 2*torch.tensor(
            [2*np.pi * i/N,
             2*np.pi * i/N,
             np.pi * i/N,
             ])
        inter.update_contacts(circle.q, max_iter=3)
        inter.vis(path, title=f"step {i}/{N}  |  ")


if __name__ == "__main__":
    main()



