"""Visualization utilities for xG analysis.

Contains small convenience plotting functions used by notebooks and the app.
Functions here intentionally return matplotlib figures so callers can further
customize rendering or save figures.
"""

import matplotlib.pyplot as plt


def plot_shot_map(df):
    """Create a simple scatter shot map.

    Expects `df` to contain `x` and `y` columns (pitch coordinates). Missing
    coordinates are handled gracefully by using empty sequences.

    Returns a matplotlib `Figure` instance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scat = ax.scatter(df.get('x', []), df.get('y', []))
    scat.set_alpha(0.6)
    ax.set_title('Shot map')
    return fig
