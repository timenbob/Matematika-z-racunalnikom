import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def F(phi,c):
    stevec = np.exp((phi/(np.pi*c)) + 1/(2*c)) - 1
    imenovalec = np.exp(1/c) - 1
    return np.pi * (stevec/imenovalec - 1/2)

def f1(phi,c): return np.sin(F(phi,c))
def f2(phi,c): return -f1(-phi,c)

def a_utez(theta,phi,c):
    f1v = f1(phi,c)
    f2v = f2(phi,c)
    stevec = np.cos(theta)**2 * (1.0 - f1v**2)
    imenovalec = stevec + np.sin(theta)**2 * (1.0 - f2v**2)
    eps = 1e-12
    # varno elementno deljenje (brez NaN/inf)
    return np.divide(stevec, imenovalec, out=np.zeros_like(stevec), where=np.abs(imenovalec) > eps)

def a_utez_2(theta,phi,c):
    return 1/(1+np.tan(theta)**2 * (np.cos(F(phi,c))**2)/(np.cos(F(-phi,c))**2))


def delta_R(theta,phi,c):
    out = np.empty_like(theta, dtype=float)
    mask_hi = np.isclose(phi,  np.pi/2)
    mask_lo = np.isclose(phi, -np.pi/2)
    mask_mid = ~(mask_hi | mask_lo)

    if np.any(mask_mid):
        aw = a_utez(theta[mask_mid], phi[mask_mid], c)
        out[mask_mid] = aw * f1(phi[mask_mid], c) + (1 - aw) * f2(phi[mask_mid], c)

    out[mask_hi] =  1.0
    out[mask_lo] = -1.0
    return out

def R(theta,phi,c,d):
    return 1 + d * delta_R(theta,phi,c)

def sfericne_kartezicne_mreza(skala,c, d, n_theta=240, n_phi=400):
    # azimut (0..2π) brez podvojitve šiva; latituda (-π/2..π/2) brez zgornjega roba
    theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    phi   = np.linspace(-np.pi/2, np.pi/2, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    RR = skala*R(TH, PH, c, d)
    # pravilna pretvorba za (azimut θ, latituda φ)
    X = RR * np.cos(PH) * np.cos(TH)
    Y = RR * np.cos(PH) * np.sin(TH)
    Z = RR * np.sin(PH)
    return X, Y, Z

def plotiranje(skala,c, d, color_by='radius'):
    X, Y, Z = sfericne_kartezicne_mreza(skala,c, d)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    if color_by == 'radius':
        RR = np.sqrt(X**2 + Y**2 + Z**2)
        norm = Normalize(vmin=np.nanmin(RR), vmax=np.nanmax(RR))
        colors = plt.cm.viridis(norm(RR))
        ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=True, shade=False)
    elif color_by == 'z':
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=True, cmap='viridis')
    else:
        ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=True)

    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Gombóc (c={c}, d={d})')
    plt.show()

# primer bolj “zdrave” vizualizacije:
plotiranje(0.0681,0.27, 0.08)
