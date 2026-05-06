import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os


# ==========================================================
# LOAD MESH
# ==========================================================

file_path = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(file_path, "plots")
os.makedirs(plots_dir, exist_ok=True)
npz_file  = os.path.join(file_path, "mesh_data.npz")

data   = np.load(npz_file, allow_pickle=True)
points = data["points"]
cells  = data["cells"].tolist()


# ==========================================================
# PARAMETRI FISICI E NUMERICI
# ==========================================================

Re     = 3200
rho    = 1.0
U_lid  = 1.0
mu     = rho * U_lid / Re

omega_u_start = 0.5
omega_u_end   = 0.7
omega_p_start = 0.3
omega_p_end   = 0.5
ramp_iters    = 100        # iterazioni di rampa per omega

max_iter = 1800+1
tol      = 1e-6            # soglia residuo normalizzato (continuità)

n_cells = len(cells)


# ==========================================================
# INIZIALIZZAZIONE CAMPI
# ==========================================================

u = np.zeros(n_cells)
v = np.zeros(n_cells)
p = np.zeros(n_cells)

u_star = np.zeros(n_cells)
v_star = np.zeros(n_cells)

ap_u = np.zeros(n_cells)
ap_v = np.zeros(n_cells)

grad_u = np.zeros((n_cells, 2))
grad_v = np.zeros((n_cells, 2))
grad_p = np.zeros((n_cells, 2))

grad_u_rec = np.zeros((n_cells, 2))
grad_v_rec = np.zeros((n_cells, 2))


# ==========================================================
# FUNZIONI
# ==========================================================

# precalcolo geometria LSQ
lsq_Minv = np.zeros((n_cells, 2, 2))   # (M^T M)^{-1} per ogni cella
lsq_MT   = []                           # M^T per ogni cella (lista di array 2×nfacce)
lsq_weights = []   # lista di array 1D, uno per cella
for cell in cells:
    P      = cell["id"]
    xP, yP = cell["centroid"]
    M_rows = []
    weights = []
    for face in cell["faces"]:
        N = face["neighbor"]
        if N is None:
            xb, yb = face["center"]
        else:
            xb, yb = cells[N]["centroid"]
        dx = xb - xP
        dy = yb - yP
        w  = 1.0 / np.sqrt(dx*dx + dy*dy)
        M_rows.append([w*dx, w*dy])
        weights.append(w)
    M    = np.array(M_rows)             # shape (nfacce, 2)
    MtM  = M.T @ M + 1e-14*np.eye(2)
    lsq_Minv[P] = np.linalg.inv(MtM)
    lsq_MT.append(M.T)                  # shape (2, nfacce)
    lsq_weights.append(np.array(weights))


def gradient_field(variable, var_moving_wall, var_fixed_wall):
    grad_out = np.zeros((n_cells, 2))
    for cell in cells:
        P = cell["id"]
        rhs = []
        for face in cell["faces"]:
            N = face["neighbor"]
            if N is None:
                if face["boundary"] == "moving_wall":
                    rhs.append(var_moving_wall[P] - variable[P])
                else:
                    rhs.append(var_fixed_wall[P]  - variable[P])
            else:
                rhs.append(variable[N] - variable[P])
        # w è già assorbita in lsq_MT[P]
        rhs = np.array(rhs) * lsq_weights[P]
        grad_out[P] = lsq_Minv[P] @ (lsq_MT[P] @ rhs)
    return grad_out


def gradient_rec(variable, var_fixed_wall, var_moving_wall, grad_variable):
    """
    Limitatore di Venkatakrishnan sul gradiente.
    """
    grad_rec_out = np.zeros((n_cells, 2))
    for cell in cells:
        P = cell["id"]

        var_max = variable[P]
        var_min = variable[P]
        for face in cell["faces"]:
            N = face["neighbor"]
            if N is not None:
                var_max = max(var_max, variable[N])
                var_min = min(var_min, variable[N])

        delta_max = var_max - variable[P]
        delta_min = var_min - variable[P]
        alpha     = None

        for face in cell["faces"]:
            N  = face["neighbor"]
            xf = face["center"]
            s  = xf - cells[P]["centroid"]

            if N is not None:
                delta_f = np.dot(s, grad_variable[P])
            else:
                if face["boundary"] == "moving_wall":
                    delta_f = var_moving_wall[P] - variable[P]
                else:
                    delta_f = var_fixed_wall[P]  - variable[P]

            if delta_f > 0:
                rf = delta_f / delta_max if delta_max != 0.0 else 1e4
            elif delta_f < 0:
                rf = delta_f / delta_min if delta_min != 0.0 else 1e4
            else:
                rf = 1e4

            af = (2.0*rf + 1.0) / (rf*(2.0*rf + 1.0) + 1.0)   # Venkatakrishnan
            alpha = af if alpha is None else min(alpha, af)

        grad_rec_out[P] = (alpha if alpha is not None else 1.0) * grad_variable[P]
    return grad_rec_out



# ==========================================================
# SETUP PLOT IN TEMPO REALE
# ==========================================================

plt.ion()

fig, (ax_vel, ax_res) = plt.subplots(
    1, 2,
    figsize=(13, 5),
    gridspec_kw={"width_ratios": [1, 1.4]}
)
fig.suptitle(f"SIMPLE solver, Re={Re}", fontsize=12, fontweight="bold")


# ---------- pannello sinistro: campo di velocità (patch) ----------
ax_vel.set_aspect("equal")
ax_vel.set_xlim(-0.01, 1.01)
ax_vel.set_ylim(-0.01, 1.01)
ax_vel.set_xlabel("x", fontsize=10);  ax_vel.set_ylabel("y", fontsize=10)
ax_vel.set_title("Iter 0 — Velocity magnitude", fontsize=11)

cmap_vel   = cm.RdYlBu_r
norm_vel   = mcolors.Normalize(vmin=0.0, vmax=1.0)

# costruisci una patch per ogni cella (lista fissa, aggiornabile via set_facecolor)
patch_list = []
for cell in cells:
    coords = np.array([points[n] for n in cell["nodes"]])
    poly   = mpatches.Polygon(coords, closed=True)
    patch_list.append(poly)

from matplotlib.collections import PatchCollection
pc_vel = PatchCollection(
    patch_list,
    cmap=cmap_vel,
    norm=norm_vel,
    edgecolors="black",
    linewidths=0.15,
    alpha=1.0
)
vel_mag_init = np.zeros(n_cells)
pc_vel.set_array(vel_mag_init)
ax_vel.add_collection(pc_vel)

cbar_vel = fig.colorbar(pc_vel, ax=ax_vel, fraction=0.046, pad=0.04)
cbar_vel.set_label("Velocity magnitude", fontsize=9)

# ---------- pannello destro: residui ----------
ax_res.set_yscale("log")
ax_res.set_xlabel("Iteration", fontsize=10)
ax_res.set_ylabel("Normalized residuals", fontsize=10)
ax_res.set_title("Residuals", fontsize=11)
ax_res.grid(True, which="both", alpha=0.35)

line_mass, = ax_res.plot([], [], color="#1D9E75", lw=1.5, label="continuity")
line_u,    = ax_res.plot([], [], color="#378ADD", lw=1.5, label="x-momentum")
line_v,    = ax_res.plot([], [], color="#E24B4A", lw=1.5, label="y-momentum")
ax_res.legend(loc="upper right", fontsize=7)

plt.tight_layout()
fig.subplots_adjust(wspace=0.35)
plt.show()

# storici residui
hist_u    = []
hist_v    = []
hist_mass = []
res_u_ref = res_v_ref = res_mass_ref = None


# ==========================================================
# LOOP SIMPLE
# ==========================================================

for it in range(max_iter):

    ramp         = min(it / max(ramp_iters, 1), 1.0)
    omega_u      = omega_u_start + (omega_u_end - omega_u_start) * ramp
    omega_p      = omega_p_start + (omega_p_end - omega_p_start) * ramp

    # ======================================================
    # 1)  GRADIENTI  (Weighted Least Squares)
    # ======================================================

    grad_u = gradient_field(u, np.ones(n_cells)*U_lid, np.zeros(n_cells))
    grad_v = gradient_field(v, np.zeros(n_cells),      np.zeros(n_cells))
    grad_p = gradient_field(p, p,                      p)               # dN/dp=0 alle pareti

    # ======================================================
    # 2)  LIMITATORE DI VENKATAKRISHNAN
    # ======================================================

    grad_u_rec = gradient_rec(u, np.zeros(n_cells), np.ones(n_cells)*U_lid, grad_u)
    grad_v_rec = gradient_rec(v, np.zeros(n_cells), np.zeros(n_cells),      grad_v)
    # gradiente di pressione NON limitato → preserva la correzione Rhie-Chow
    grad_p_rec = grad_p.copy()

    # ======================================================
    # 3)  ASSEMBLAGGIO EQUAZIONI DI QUANTITÀ DI MOTO
    #
    #     Schema: Upwind 2° ordine (SOU)
    #       termine implicito:  UDS (1° ordine) → matrice
    #       deferred correction: grad·dx aggiunto al RHS (esplicito)
    #     Diffusione: schema centrale (mesh ortogonale)
    # ======================================================

    Au = lil_matrix((n_cells, n_cells))
    Av = lil_matrix((n_cells, n_cells))
    bu = np.zeros(n_cells)
    bv = np.zeros(n_cells)

    for cell in cells:
        P = cell["id"]
        for face in cell["faces"]:
            Sf     = face["length"]
            nx, ny = face["normal"]
            N      = face["neighbor"]

            # --------------------------------------------------
            # FACCE INTERNE
            # --------------------------------------------------
            if N is not None:
                d  = face["distance"]
                xf = face["center"]

                # Flusso convettivo stimato con media lineare (per upwinding)
                u_mid = 0.5*(u[P] + u[N])
                v_mid = 0.5*(v[P] + v[N])
                F     = rho * (u_mid*nx + v_mid*ny) * Sf   # [kg/s]

                # Coefficiente diffusivo (mesh ortogonale)
                D = mu * Sf / d

                # Cella upwind (determinata UNA VOLTA dal flusso lineare)
                up = P if F >= 0.0 else N

                # ---- Implicito: UDS puro ----
                #   ap_P += max(F,0) + D
                #   an_N  = min(F,0) - D
                Au[P, P] += max(F, 0.0) + D
                Av[P, P] += max(F, 0.0) + D
                Au[P, N] += min(F, 0.0) - D
                Av[P, N] += min(F, 0.0) - D

                # ---- Esplicito: deferred correction SOU ----
                #   contributo = F · (grad_up · dx_up)
                #   dove dx_up = xf - x_up  (vettore da upwind a faccia)
                dx_up = xf - cells[up]["centroid"]
                bu[P] -= F * np.dot(grad_u_rec[up], dx_up)
                bv[P] -= F * np.dot(grad_v_rec[up], dx_up)

            # --------------------------------------------------
            # FACCE AL CONTORNO  (solo diffusione, no convezione)
            # --------------------------------------------------
            else:
                bc = face["boundary"]
                if bc == "fixed_wall":
                    u_wall, v_wall = 0.0, 0.0
                elif bc == "moving_wall":
                    u_wall, v_wall = U_lid, 0.0
                else:
                    u_wall, v_wall = 0.0, 0.0

                d = np.linalg.norm(face["center"] - cell["centroid"])
                D = mu * Sf / d
                Au[P, P] += D
                bu[P]    += D * u_wall
                Av[P, P] += D
                bv[P]    += D * v_wall
                
            # ----------------------------------
            # Pressione
            # ----------------------------------
            
            if N is not None:
                # Interpolazione lineare della pressione alla faccia
                pf = 0.5 * p[P] + 0.5 * p[N]
            else:
                # Alle pareti: pressione della cella (gradiente normale nullo)
                pf = p[P]

            bu[P] -= pf * Sf * nx
            bv[P] -= pf * Sf * ny

    # ======================================================
    # 4)  TERMINE GRADIENTE DI PRESSIONE  (→ RHS)
    #     Interpolazione lineare della pressione alla faccia
    # ======================================================

    ap_u = Au.diagonal().copy()   # salva PRIMA di modificare la diagonale
    ap_v = Av.diagonal().copy()


    # ======================================================
    # 5)  RESIDUO DI MOMENTUM  (calcolato prima di modificare la matrice)
    # ======================================================

    

    # ======================================================
    # 6)  UNDER-RELAXATION + SOLUZIONE  u*, v*
    # ======================================================

    Au.setdiag(ap_u / omega_u)
    bu += (ap_u / omega_u - ap_u) * u

    Av.setdiag(ap_v / omega_u)
    bv += (ap_v / omega_u - ap_v) * v

    u_star = spsolve(Au.tocsr(), bu)
    v_star = spsolve(Av.tocsr(), bv)

    # ======================================================
    # 7)  RHIE-CHOW: flusso di massa corretto alle facce interne
    #
    #  Formula scalare (proiettata su n̂):
    #
    #   F*_f = [ ū*_f·n̂  −  d̄_f·( (pN−pP)/d  −  (∇p·n̂)_f_interp ) ] · Sf
    #
    #   d̄_f = 0.5·(VP/ap_P + VN/ap_N)   [m²·s/kg]
    #   (pN−pP)/d              = derivata direzionale compatta (ortogonale)
    #   (∇p·n̂)_f_interp       = media delle proiezioni nodali su n̂
    # ======================================================

    Ap_prime = lil_matrix((n_cells, n_cells))
    bp_prime = np.zeros(n_cells)

    for cell in cells:
        P = cell["id"]
        area_P = cell["area"]

        for face in cell["faces"]:
            N = face["neighbor"]
            if N is None:
                continue

            Sf = face["length"]
            nx, ny = face["normal"]
            d = face["distance"]
            area_N = cells[N]["area"]

            # --- Velocità normale interpolata ---
            u_f_bar = 0.5 * u_star[P] + 0.5 * u_star[N]
            v_f_bar = 0.5 * v_star[P] + 0.5 * v_star[N]
            flux_vel_bar = u_f_bar * nx + v_f_bar * ny  # [m/s]

            # --- Coefficiente d̄_f (Rhie-Chow) ---
            dP = area_P / ap_u[P]   # usa ap originale (senza omega)
            dN = area_N / ap_u[N]
            df_bar = 0.5 * (dP + dN)   # [m²·s/kg]

            # --- Gradiente di pressione compatto (diretto, lungo n̂) ---
            grad_p_direct = (p[N] - p[P]) / d   # [Pa/m]

            # --- Gradiente di pressione interpolato proiettato su n̂ ---
            grad_p_P_normal = grad_p[P, 0] * nx + grad_p[P, 1] * ny
            grad_p_N_normal = grad_p[N, 0] * nx + grad_p[N, 1] * ny
            grad_p_interp_normal = 0.5 * (grad_p_P_normal + grad_p_N_normal)

            # --- Flusso Rhie-Chow ---
            # F*_f = [ū_f·n̂ - d̄_f·(∇p_direct - ∇p_interp)] · Sf
            flux_rc = (flux_vel_bar - df_bar * (grad_p_direct - grad_p_interp_normal)) * Sf
            F_star = rho * flux_rc   # [kg/s]

            # --- Coefficiente per equazione p' ---
            # Qp = rho · df_bar · Sf / d   (Laplaciano di p')
            Qp = rho * df_bar * Sf / d

            #var_p_prime.append((P, N, Qp, F_star))
            Ap_prime[P, P] += Qp
            Ap_prime[P, N] -= Qp
            bp_prime[P] -= F_star

    # ======================================================
    # 8)  EQUAZIONE PRESSIONE CORRETTIVA
    #     ∑_f Qp·(p'_N − p'_P) = − ∑_f F*_f
    # ======================================================


    # Pin pressione di riferimento in cella 0 → sistema non singolare
    Ap_prime[0, :] = 0.0
    Ap_prime[0, 0] = 1.0
    bp_prime[0]    = 0.0

    # ======================================================
    # 9)  SOLUZIONE p'
    # ======================================================

    p_prime  = spsolve(Ap_prime.tocsr(), bp_prime)
    p_prime -= np.mean(p_prime)    # rimuovi deriva numerica (modo nullo)

    # ======================================================
    # 10) GRADIENTE DI p'
    # ======================================================

    grad_p_prime = gradient_field(p_prime, p_prime, p_prime)

    # ======================================================
    # 11) AGGIORNAMENTO PRESSIONE E VELOCITÀ
    # ======================================================

    p += omega_p * p_prime

    for cell in cells:
        P = cell["id"]
        V = cell["area"]
        u[P] = u_star[P] - grad_p_prime[P, 0] * V / ap_u[P]
        v[P] = v_star[P] - grad_p_prime[P, 1] * V / ap_v[P]

    # ======================================================
    # 12) RESIDUI  (continuità post-correzione)
    # ======================================================

    #r_mass_raw = compute_mass_residual(u, v)
    Au_csr = Au.tocsr()
    Av_csr = Av.tocsr()

    r_u_raw    = np.sum(np.abs(bu - Au_csr.dot(u)))
    r_v_raw    = np.sum(np.abs(bv - Av_csr.dot(v)))
    r_mass_raw = np.linalg.norm(bp_prime)

    # Normalizzazione alla prima iterazione
    if it == 0:
        res_u_ref    = r_u_raw    if r_u_raw    > 0.0 else 1.0
        res_v_ref    = r_v_raw    if r_v_raw    > 0.0 else 1.0
        res_mass_ref = r_mass_raw if r_mass_raw > 0.0 else 1.0

    r_u_norm    = r_u_raw    / res_u_ref
    r_v_norm    = r_v_raw    / res_v_ref
    r_mass_norm = r_mass_raw / res_mass_ref

    hist_u.append(r_u_norm)
    hist_v.append(r_v_norm)
    hist_mass.append(r_mass_norm)

    # Stampa a console
    if it % 10 == 0:
        print(f"Iter {it:4d} | continuity: {r_mass_norm:.3e} | x-mom: {r_u_norm:.3e} | "f"y-mom: {r_v_norm:.3e} ")

    # ======================================================
    # 13) PLOT IN TEMPO REALE  (ogni 10 iterazioni)
    # ======================================================

    if it % 10 == 0:
        # Mappa velocità
        # --- campo di velocità: aggiorna colori delle patch ---
        vel_mag = np.sqrt(u**2 + v**2)
        vmax    = float(np.max(vel_mag)) if np.max(vel_mag) > 0 else 1.0
        pc_vel.set_array(vel_mag)
        pc_vel.set_clim(0.0, vmax)
        norm_vel.vmax = vmax
        ax_vel.set_title(f"Iter {it} — Velocity magnitude", fontsize=11)

        # --- residui ---
        iters = np.arange(len(hist_u))
        line_u.set_data(iters,    hist_u)
        line_v.set_data(iters,    hist_v)
        line_mass.set_data(iters, hist_mass)
        ax_res.relim()
        ax_res.autoscale_view()

        fig.canvas.draw_idle()
        plt.pause(0.01)
    # ======================================================
    # 14) CRITERIO DI CONVERGENZA
    # ======================================================

    if r_mass_norm < tol:
        print(f"\nStopping criteria satisfied at iteration {it}")
        break

else:
    print(f"\n Maximum iteration criteria satisfied at iteration{max_iter}.")

plt.ioff()

# Salva plot residui
res_file = os.path.join(plots_dir, f"Re{Re}.png")
fig.savefig(res_file, dpi=450)
#print(f"Plot residui salvato in: {os.path.join(plots_dir, f'residuals_Re{Re}.png')}")


# ==========================================================
# POST-PROCESSING: confronto con Ghia et al. (1982)
# ==========================================================

# --- v(x) lungo y = 0.5 ---
tol_geom = 1e-3
x_line, v_line = [], []
for cell in cells:
    x, y = cell["centroid"]
    if abs(y - 0.5) < tol_geom:
        x_line.append(x)
        v_line.append(v[cell["id"]])
x_line = np.array(x_line);  v_line = np.array(v_line)
idx    = np.argsort(x_line); x_line = x_line[idx]; v_line = v_line[idx]

# --- u(y) lungo x = 0.5 ---
y_vert, u_vert = [], []
for cell in cells:
    x, y = cell["centroid"]
    if abs(x - 0.5) < tol_geom:
        y_vert.append(y)
        u_vert.append(u[cell["id"]])
y_vert = np.array(y_vert);  u_vert = np.array(u_vert)
idx2   = np.argsort(y_vert); y_vert = y_vert[idx2]; u_vert = u_vert[idx2]

# Dati Ghia et al. 1982
x_ghia = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
    0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
    0.0703, 0.0625, 0.0000
])
v_ghia_dict = {
    100:  np.array([ 0.00000, -0.05906, -0.07391, -0.08864, -0.10133, -0.16914,
                    -0.22445, -0.24533,  0.05454,  0.17527,  0.17507,  0.16077,
                    0.12317,  0.10890,  0.10091,  0.09233,  0.00000]),
    400:  np.array([ 0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
                    -0.44993, -0.38598,  0.05186,  0.30174,  0.30203,  0.28124,
                    0.22965,  0.20920,  0.19713,  0.18360,  0.00000]),
    1000: np.array([ 0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                    -0.42665, -0.31966,  0.02526,  0.32235,  0.33075,  0.37095,
                    0.32627,  0.30353,  0.29012,  0.27485,  0.00000]),
    3200: np.array([ 0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307,
                    -0.37401, -0.31184,  0.00999,  0.28188,  0.29030,  0.37119,
                    0.42768,  0.41906,  0.40917,  0.39560,  0.00000]),
    5000: np.array([ 0.00000, -0.49774, -0.55069, -0.55408, -0.52876, -0.41442,
                    -0.36214, -0.30018,  0.00945,  0.27280,  0.28066,  0.35368,
                    0.42951,  0.43648,  0.43329,  0.42447,  0.00000]),
    7500: np.array([ 0.00000, -0.53858, -0.55216, -0.52347, -0.48590, -0.41050,
                    -0.36213, -0.30448,  0.00824,  0.27348,  0.28117,  0.35060,
                    0.41824,  0.43564,  0.44030,  0.43979,  0.00000]),
    10000:np.array([ 0.00000, -0.54302, -0.52987, -0.49099, -0.45863, -0.41496,
                    -0.36737, -0.30719,  0.00831,  0.27224,  0.28003,  0.35070,
                    0.41487,  0.43124,  0.43733,  0.43983,  0.00000]),
}
y_ghia = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
])
u_ghia_dict = {
    100:  np.array([ 1.00000,  0.84123,  0.78871,  0.73722,  0.68717,  0.23151,  0.00332,
                    -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
                    -0.04775, -0.04192, -0.03717,  0.00000]),
    400:  np.array([ 1.00000,  0.75837,  0.68439,  0.61756,  0.55892,  0.29093,  0.16256,
                    0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
                    -0.10338, -0.09266, -0.08186,  0.00000]),
    1000: np.array([ 1.00000,  0.65928,  0.57492,  0.51117,  0.46604,  0.33304,  0.18719,
                    0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
                    -0.22220, -0.20196, -0.18109,  0.00000]),
    3200: np.array([ 1.00000,  0.53236,  0.48296,  0.46547,  0.46101,  0.34682,  0.19791,
                    0.07156, -0.04272, -0.08664, -0.24427, -0.34323, -0.41933,
                    -0.37827, -0.35344, -0.32407,  0.00000]),
    5000: np.array([ 1.00000,  0.48223,  0.46120,  0.45992,  0.46036,  0.33556,  0.20087,
                    0.08183, -0.03039, -0.07404, -0.22847, -0.33050, -0.40435,
                    -0.43643, -0.42901, -0.41165,  0.00000]),
    7500: np.array([ 1.00000,  0.47244,  0.47048,  0.47323,  0.47167,  0.34228,  0.20591,
                    0.08342, -0.03800, -0.07503, -0.23176, -0.32393, -0.38324,
                    -0.43025, -0.43590, -0.43154,  0.00000]),
    10000:np.array([ 1.00000,  0.47221,  0.47783,  0.48070,  0.47804,  0.34635,  0.20673,
                    0.08344, -0.03744, -0.07507, -0.23186, -0.32709, -0.38000,
                    -0.41657, -0.42537, -0.42735,  0.00000]),
}

v_ghia_ref = v_ghia_dict.get(Re)
u_ghia_ref = u_ghia_dict.get(Re)



from matplotlib.ticker import MultipleLocator

fig_cmp, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig_cmp.suptitle(f"Lid-Driven Cavity, Re={Re} — Comparison with Ghia et al. (1982)",
                fontsize=12, fontweight="bold")

# v(x) @ y=0.5
ax1.plot(x_line, v_line, "-", lw=2, color="#E24B4A", label="This work")
if v_ghia_ref is not None:
    ax1.plot(x_ghia, v_ghia_ref, "^k", ms=6, label="Ghia et al.")
ax1.set_xlabel("x"); ax1.set_ylabel("v")
ax1.set_title("v(x)  @ y = 0.5", fontsize=11)
ax1.legend()

ax1.minorticks_on()
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
ax1.grid(True, which='major', linewidth=0.8, alpha=0.4)
ax1.grid(True, which='minor', linewidth=0.5, alpha=0.2)

# u(y) @ x=0.5
ax2.plot(u_vert, y_vert, "-", lw=2, color="#E24B4A", label="This work")
if u_ghia_ref is not None:
    ax2.plot(u_ghia_ref, y_ghia, "^k", ms=6, label="Ghia et al.")
ax2.set_xlabel("u"); ax2.set_ylabel("y")
ax2.set_title("u(y)  @ x = 0.5", fontsize=11)
ax2.legend()

ax2.minorticks_on()
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
ax2.grid(True, which='major', linewidth=0.8, alpha=0.4)
ax2.grid(True, which='minor', linewidth=0.5, alpha=0.2)

plt.tight_layout()
cmp_file = os.path.join(plots_dir, f"comparison_Re{Re}.png")
fig_cmp.savefig(cmp_file, dpi=250)

plt.show(block=True)