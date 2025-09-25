# cooling_conduction_convection.py
# Forced-convection + multilayer-conduction calculator (planar 1D)
# Edit the USER INPUT section and run with:  python cooling_conduction_convection.py
#
# Outputs:
# - h (W/m²·K), Re, Pr, Nu
# - Layer-by-layer interface temperatures (°C)
# - Water outlet temperature and bulk average
# - Velocity sweep helper
#
# Notes:
# • If you need cylindrical conduction (pipe wall), extend with the cylindrical formula;
#   this script uses planar 1D to match the original approach.
# • Thickness in meters, conductivity in W/m·K, area in m².

import math
import numpy as np

# =========================
# ======= USER INPUT ======
# =========================

# Heat input from one coil [W]
q = 20000 / 48

# Copper (coil) hot-side maximum temperature [°C]
T_copper_hot_C = 85.0

# Water inlet bulk temperature [°C]
T_water_in_C = 20.0

# Pipe inner diameter [m]  (water flows inside this)
D_inner = 0.010  # 1 cm

# Mean water velocity [m/s] inside the pipe
u = 5.0

# Exposed heat transfer area for ONE coil path [m²]
# Example: 0.09 m × 0.123 m
A = 0.09 * 0.123

# Layer stack, from HOT (copper side) to COLD (water side):
# [thickness (m), conductivity k (W/m·K), name]
layers = np.array([
    [3.35e-3, 401.0, 'Copper'],       # coils (through-thickness path)
    [1.00e-3,   3.7, 'Epoxy/Encaps.'],
    [0.50e-3, 237.0, 'Aluminium']     # pipe wall (water contacts the cold side of this)
], dtype=object)

# Fluid properties for water (approx. 20–40 °C)
rho = 997.0           # kg/m³
mu = 0.0010           # Pa·s
cp = 4180.0           # J/kg·K
k_water = 0.6         # W/m·K

# Dittus–Boelter exponent n (0.4 heating fluid; 0.3 cooling fluid)
n_db = 0.4

# =========================
# ====== CALCULATIONS =====
# =========================

def K_to_C(TK): 
    return TK - 273.15

# Convert temps to K
T_copper_hot_K = T_copper_hot_C + 273.15
T_water_in_K = T_water_in_C + 273.15

# Flow area (circular)
A_flow = math.pi * (D_inner**2) / 4.0

# Mass flow
m_dot = rho * A_flow * u

# Dimensionless groups
Re = rho * u * D_inner / mu
Pr = cp * mu / k_water

# Nusselt
if Re < 2300:
    Nu = 3.66  # laminar fallback
else:
    Nu = 0.023 * (Re**0.8) * (Pr**n_db)

# Convective coefficient
h = Nu * k_water / D_inner

# Thermal resistances
R_layers = [(float(L) / (float(k) * A)) for (L, k, _) in layers]
R_cond = sum(R_layers)
R_conv = 1.0 / (h * A)
R_total = R_cond + R_conv

# Water temperature rise
dT_water = q / (m_dot * cp) if m_dot > 0 else float('inf')
T_water_out_K = T_water_in_K + dT_water
T_water_bulk_avg_K = 0.5 * (T_water_in_K + T_water_out_K)

# Interface temperatures (hot → cold)
T_interfaces_K = [T_copper_hot_K]
for R in R_layers:
    T_interfaces_K.append(T_interfaces_K[-1] - q * R)

T_wall_water_side_K = T_interfaces_K[-1]
T_water_bulk_K = T_wall_water_side_K - q * R_conv

# =========================
# ========= I/O ===========
# =========================

print("=== Forced Convection + Multilayer Conduction (Planar 1D) ===")
print(f"Heat load q                       : {q:,.2f} W")
print(f"Water velocity u                  : {u:.3f} m/s")
print(f"Pipe inner diameter D            : {D_inner*1000:.1f} mm")
print(f"Flow area A_flow                 : {A_flow*1e6:.2f} cm²")
print(f"Mass flow m_dot                  : {m_dot:.4f} kg/s")
print(f"Reynolds number Re               : {Re:,.0f}")
print(f"Prandtl number Pr                : {Pr:.2f}")
print(f"Nusselt Nu                       : {Nu:.1f}")
print(f"h (convective coeff.)            : {h:,.0f} W/m²·K\n")

print(f"Heat transfer area A             : {A:.5f} m²")
print("Layer stack (hot → cold):")
for (L, k, name), R in zip(layers, R_layers):
    print(f"  - {name:<16}  L={float(L)*1e3:6.2f} mm, k={float(k):7.2f} W/m·K  => R = {R:.6e} K/W")
print(f"Conduction resistance R_cond     : {R_cond:.6e} K/W")
print(f"Convection resistance R_conv     : {R_conv:.6e} K/W")
print(f"Total resistance R_total         : {R_total:.6e} K/W\n")

print(f"Hot copper temperature           : {T_copper_hot_C:.2f} °C")
print(f"Water inlet temperature          : {T_water_in_C:.2f} °C")
print(f"Predicted water ΔT               : {dT_water:+.3f} K")
print(f"Water outlet temperature         : {K_to_C(T_water_out_K):.2f} °C")
print(f"Water bulk (avg) temperature     : {K_to_C(T_water_bulk_avg_K):.2f} °C\n")

print("Interface temperatures (°C), hot → cold:")
for (L,k,name), T in zip(layers, T_interfaces_K[1:]):
    print(f"  after {name:<16}: {K_to_C(T):.2f} °C")
print(f"Aluminium (water-side) wall temp : {K_to_C(T_wall_water_side_K):.2f} °C")
print(f"Bulk water temperature (near wall): {K_to_C(T_water_bulk_K):.2f} °C")

def quick_sweep_velocities(u_list):
    """
    Return a list of (u, h, T_wall_water_side_C, T_bulk_water_C) for quick inspection.
    """
    out = []
    for u_i in u_list:
        Re_i = rho * u_i * D_inner / mu
        if Re_i < 2300:
            Nu_i = 3.66
        else:
            Nu_i = 0.023 * (Re_i**0.8) * (Pr**n_db)
        h_i = Nu_i * k_water / D_inner
        R_conv_i = 1.0 / (h_i * A)
        T_wall_i = T_copper_hot_K - q * R_cond
        T_wbulk_i = T_wall_i - q * R_conv_i
        out.append((u_i, h_i, K_to_C(T_wall_i), K_to_C(T_wbulk_i)))
    return out

if __name__ == "__main__":
    u_vals = [1, 2, 3, 4, 5, 6]
    sweep = quick_sweep_velocities(u_vals)
    print("\nVelocity sweep (u, h, T_wall_water_side, T_bulk_water):")
    for row in sweep:
        print(f"  u={row[0]:.1f} m/s  |  h={row[1]:8.0f} W/m²K  |  T_wall={row[2]:6.2f} °C  |  T_bulk={row[3]:6.2f} °C")
