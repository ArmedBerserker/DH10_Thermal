# cooling_conduction_convection_merged.py
# Combined conduction + convection calculator with TWO models:
# 1) Lumped-q model (you provide total heat q from one coil)  -> quick sizing
# 2) NTU distributed model (self-consistent wall temperature)  -> correct outlet
#
# Edit USER INPUT and run:  python cooling_conduction_convection_merged.py
# Requires: numpy, matplotlib (for plots)
#
# Outputs:
# - h, Re, Pr, Nu
# - Interface temperatures (hot → cold)
# - Lumped model: uses provided q to compute drops and an approximate "local bulk" (not outlet)
# - NTU model: solves Tw so conduction equals convection; returns Q, T_out, T_avg
# - Two plots: h vs velocity; T_out vs velocity (NTU)
#
# Notes:
# • Solid conduction is planar 1D through layers, matching the user's original approach.
# • NTU uses A_wet = P*L with P = π D_inner. If you prefer to enter L directly, set A_wet = π D * L.
# • Thickness [m], conductivity [W/m·K], area [m²].

import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ======= USER INPUT ======
# =========================

# HEAT INPUT (for lumped model only)
q = 20000 / 48        # W  (total heat from one coil)

# TEMPERATURES
T_copper_hot_C = 85.0 # °C (hot copper / coil)
T_water_in_C   = 20.0 # °C (inlet bulk)

# GEOMETRY & FLOW
D_inner = 0.010       # m (pipe inner diameter)
u       = 5.0         # m/s (mean velocity inside pipe)

# CONVECTION WETTED AREA (A_wet = P * L). If you know length L, set A_wet = math.pi*D_inner*L
A_wet = 0.09 * 0.123  # m² (your aluminium-water contact area per coil)

# LAYER STACK, hot → cold
# [thickness (m), conductivity k (W/m·K), name]
layers = np.array([
    [3.35e-3, 401.0, 'Copper'],
    [1.00e-3,   3.7, 'Epoxy/Encaps.'],
    [0.50e-3, 237.0, 'Aluminium']
], dtype=object)

# WATER PROPERTIES (approx. 20–40 °C)
rho      = 997.0        # kg/m³
mu       = 0.0010       # Pa·s
cp       = 4180.0       # J/kg·K
k_water  = 0.6          # W/m·K
n_db     = 0.4          # Dittus–Boelter exponent (heating the fluid)

# =========================
# ====== CALCULATIONS =====
# =========================

def K2C(TK): return TK - 273.15
def C2K(TC): return TC + 273.15

T_hot_K  = C2K(T_copper_hot_C)
T_in_K   = C2K(T_water_in_C)

# Flow area & mass flow
A_flow = math.pi * (D_inner**2) / 4.0
m_dot  = rho * A_flow * u

# Dimensionless groups & h
Re = rho * u * D_inner / mu
Pr = cp * mu / k_water
Nu = 3.66 if Re < 2300 else 0.023 * (Re**0.8) * (Pr**n_db)
h  = Nu * k_water / D_inner

# Perimeter and length from A_wet
P = math.pi * D_inner
L = A_wet / P

# Conduction resistance through solids (planar)
R_layers = [(float(Li) / (float(ki) * A_wet)) for (Li, ki, _) in layers]
R_cond   = sum(R_layers)

# ---------------------------
# (1) LUMPED-q MODEL (quick)
# ---------------------------
R_conv_lumped = 1.0 / (h * A_wet) if h > 0 else float('inf')
# Interface temperatures using provided q (hot → cold)
T_interfaces_q = [T_hot_K]
for R in R_layers:
    T_interfaces_q.append(T_interfaces_q[-1] - q * R)
T_wall_q_K = T_interfaces_q[-1]               # aluminium water-side
T_bulk_local_q_K = T_wall_q_K - q * R_conv_lumped  # local "bulk" via single resistance (not outlet)

# ---------------------------
# (2) NTU DISTRIBUTED MODEL
# ---------------------------
NTU = h * A_wet / (m_dot * cp) if m_dot > 0 else float('inf')

def f_Tw(Tw):
    # Energy balance: (T_hot - Tw)/R_cond = m_dot*cp * (Tw - T_in) * (1 - exp(-NTU))
    left  = (T_hot_K - Tw) / R_cond
    right = m_dot * cp * (Tw - T_in_K) * (1.0 - math.exp(-NTU))
    return left - right

# Solve Tw by bisection
low, high = T_in_K, T_hot_K
for _ in range(200):
    mid = 0.5 * (low + high)
    val = f_Tw(mid)
    if abs(val) < 1e-8:
        Tw = mid
        break
    if f_Tw(low) * val <= 0:
        high = mid
    else:
        low = mid
else:
    Tw = mid  # best effort

# Total heat and outlet temperature
Q_NTU = (T_hot_K - Tw) / R_cond
T_out_NTU_K = T_in_K + Q_NTU / (m_dot * cp) if m_dot > 0 else float('nan')

# Bulk-average along length (exact for constant Tw)
if NTU > 0:
    T_avg_NTU_K = T_in_K + (T_out_NTU_K - T_in_K) * (1.0 - (1.0 - math.exp(-NTU)) / NTU)
else:
    T_avg_NTU_K = T_in_K

# Interface temps with NTU heat flow
T_interfaces_NTU = [T_hot_K]
for R in R_layers:
    T_interfaces_NTU.append(T_interfaces_NTU[-1] - Q_NTU * R)
T_wall_NTU_K = T_interfaces_NTU[-1]  # should equal Tw

# =========================
# ========= OUTPUT =========
# =========================

print("=== Geometry & Flow ===")
print(f"D_inner                     : {D_inner*1000:.1f} mm")
print(f"Length from area (L)        : {L:.3f} m (A_wet = {A_wet:.4f} m²)")
print(f"Velocity u                  : {u:.3f} m/s")
print(f"m_dot                       : {m_dot:.4f} kg/s")
print(f"Re, Pr, Nu                  : {Re:,.0f}, {Pr:.2f}, {Nu:.1f}")
print(f"h                           : {h:,.0f} W/m²·K")
print(f"NTU = hA/(m_dot*cp)         : {NTU:.3f}\n")

print("=== Layers (hot → cold) ===")
for (Li, ki, name), R in zip(layers, R_layers):
    print(f"  - {name:<16} L={float(Li)*1e3:6.2f} mm, k={float(ki):6.2f} W/m·K  => R = {R:.6e} K/W")
print(f"R_cond (solids)             : {R_cond:.6e} K/W\n")

print("=== (1) Lumped-q Model (q provided) ===")
print(f"q (provided)                : {q:,.2f} W")
print(f"Wall (water-side) temp      : {K2C(T_wall_q_K):.2f} °C")
print(f"Local bulk via q*R_conv     : {K2C(T_bulk_local_q_K):.2f} °C  (NOT the true outlet)")
print(f"R_conv (1/hA)               : {R_conv_lumped:.6e} K/W\n")

print("=== (2) NTU Distributed Model (solves Tw, then T_out) ===")
print(f"Tw (solved wall temp)       : {K2C(Tw):.2f} °C")
print(f"Q_NTU (from conduction)     : {Q_NTU:,.2f} W")
print(f"T_out (true outlet)         : {K2C(T_out_NTU_K):.2f} °C")
print(f"T_avg (bulk average)        : {K2C(T_avg_NTU_K):.2f} °C\n")

print("Interface temps (°C), hot → cold [NTU heat flow]:")
for (Li, ki, name), T in zip(layers, T_interfaces_NTU[1:]):
    print(f"  after {name:<16}: {K2C(T):.2f}")
print(f"Aluminium wall (water-side) : {K2C(T_wall_NTU_K):.2f} °C (≈ Tw)")

# =========================
# ========= PLOTS =========
# =========================
# 1) h vs velocity
u_vals = np.linspace(0.5, 8.0, 40)
def compute_h(u_i):
    Re_i = rho * u_i * D_inner / mu
    Pr_i = Pr  # approx const
    Nu_i = 3.66 if Re_i < 2300 else 0.023 * (Re_i**0.8) * (Pr_i**0.4)
    return Nu_i * k_water / D_inner

h_vals = np.array([compute_h(ui) for ui in u_vals])
plt.figure()
plt.plot(u_vals, h_vals)
plt.xlabel("Water velocity u [m/s]")
plt.ylabel("Convective coefficient h [W/m²·K]")
plt.title("Water-side h vs. velocity (D=1 cm)")
plt.tight_layout()
plt.savefig("plot_h_vs_velocity.png", dpi=140)
plt.close()

# 2) NTU T_out vs velocity
def NTU_out(u_i):
    A_flow_i = math.pi * (D_inner**2) / 4.0
    m_dot_i  = rho * A_flow_i * u_i
    h_i      = compute_h(u_i)
    NTU_i    = h_i * A_wet / (m_dot_i * cp) if m_dot_i > 0 else np.inf

    # Solve Tw for each u_i
    def f_Tw_i(Tw_i):
        left  = (T_hot_K - Tw_i) / R_cond
        right = m_dot_i * cp * (Tw_i - T_in_K) * (1.0 - math.exp(-NTU_i))
        return left - right

    lo, hi = T_in_K, T_hot_K
    for _ in range(120):
        mid = 0.5*(lo+hi)
        val = f_Tw_i(mid)
        if abs(val) < 1e-7:
            Tw_i = mid
            break
        if f_Tw_i(lo)*val <= 0: hi = mid
        else: lo = mid
    else:
        Tw_i = mid

    Q_i = (T_hot_K - Tw_i) / R_cond
    T_out_i = T_in_K + Q_i / (m_dot_i * cp)
    return T_out_i

T_out_vals = np.array([NTU_out(ui) for ui in u_vals])
plt.figure()
plt.plot(u_vals, [K2C(T) for T in T_out_vals])
plt.xlabel("Water velocity u [m/s]")
plt.ylabel("Outlet temperature [°C]")
plt.title("True outlet T_out vs. velocity (NTU, self-consistent Tw)")
plt.tight_layout()
plt.savefig("plot_Tout_vs_velocity.png", dpi=140)
plt.close()

print("\nSaved plots: plot_h_vs_velocity.png, plot_Tout_vs_velocity.png")
