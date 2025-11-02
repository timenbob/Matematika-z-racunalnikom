import numpy as np
import matplotlib.pyplot as plt

def F(phi,c):
    stevec=np.exp((phi/(np.pi*c)) + 1/(2*c))-1
    imenovalec=np.exp(1/c)-1
    return np.pi *(stevec/imenovalec - 1/2)

def f1(phi,c):
    return np.sin(F(phi,c))

def f2(phi,c):
    return -f1(-phi,c)

def a_utez(theta,phi,c):
    f1v=f1(phi,c)
    f2v=f2(phi,c)
    stevec = np.cos(theta)**2 * (1.0 - f1v**2)
    imenovalec = stevec + np.sin(theta)**2 * (1.0 - f2v**2)
    return stevec/imenovalec

def delta_R(theta,phi,c):
    '''if abs(phi) < np.pi/2:
        return a_utez(theta,phi,c)*f1(phi,c)+(1-a_utez(theta,phi,c))*f2(phi,c)
    elif phi == np.pi/2:
        return 1
    elif phi == -np.pi/2: 
        return -1
    else: raise ValueError("phi must be in the interval [-pi/2, pi/2]")'''
    # osnovni izraz
    out = a_utez(theta, phi, c)*f1(phi, c) + (1 - a_utez(theta, phi, c))*f2(phi, c)
    # robni pogoji pri |phi| = pi/2 (če jih želiš)
    out = np.where(np.isclose(phi,  np.pi/2),  1.0, out)
    out = np.where(np.isclose(phi, -np.pi/2), -1.0, out)
    return out

def R(theta,phi,c,d):
    return (1+d)*delta_R(theta,phi,c)

def sfericne_kartezicne_mreza(c, d, n_theta=240, n_phi=240):
    theta = np.linspace(0.0,2*np.pi, n_theta)
    phi   = np.linspace(-np.pi/2, np.pi/2, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    RR = R(TH, PH, c, d)
    X = RR * np.cos(TH) * np.cos(PH)
    Y = RR * np.sin(TH) * np.cos(PH)
    Z = RR * np.sin(PH)
    return X, Y, Z

from matplotlib.colors import Normalize

def plotiranje(c, d, color_by='radius'):
    X, Y, Z = sfericne_kartezicne_mreza(c, d)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    if color_by == 'radius':
        RR = np.sqrt(X**2 + Y**2 + Z**2)                # skalar za barvanje
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

plotiranje(0.5,0.5)