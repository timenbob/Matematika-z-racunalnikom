import numpy as np
import plotly.graph_objects as go

# right now things don't line up at the 0,2pi interface
def radius(theta, phi, wrap=1.0, lump_width=np.pi, lump_offset=0.0, r0=1.0, dr=0.2):
    # whenever theta = c*phi, r should be a maximum (derivative is zero)
    return dr*np.cos(np.pi/2*((theta + lump_offset) - wrap*phi)/lump_width) + r0 #cos (periodic)
    # return dr*np.exp(-(theta+lump_offset - wrap*phi)**2/(lump_width**2)) + r0 # gaussian (single lump)


# Generate spherical coordinates
theta_vals = np.linspace(-np.pi/2, np.pi/2, 50)
phi_vals = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta_vals, phi_vals)


def generate_plot():
    r = radius(theta, phi)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis')])
    fig.update_layout(title='Gomboc-Like Surface', autosize=True)
    return fig

# Create an interactive figure with sliders
fig = generate_plot()
# fig.update_layout(
#     updatemenus=[
#         {
#             "buttons": [
#                 {"args": ["R0", 0.5], "label": "R0 = 0.5", "method": "relayout"},
#                 {"args": ["R0", 2.0], "label": "R0 = 2.0", "method": "relayout"}
#             ],
#             "direction": "down",
#             "showactive": True
#         }
#     ],
#     sliders=[
#         {
#             "currentvalue": {"prefix": "R0: "},
#             "steps": [
#                 {"args": [[f"R0={v}"], {"frame": {"duration": 0, "redraw": True}}], "label": f"{v}", "method": "animate"}
#                 for v in np.linspace(0.5, 2.0, 10)
#             ]
#         }
#     ]
# )

# Show the interactive plot
fig.show()
