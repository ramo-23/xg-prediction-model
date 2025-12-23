"""Visualization utilities (mplsoccer, plotly)."""
import matplotlib.pyplot as plt


def plot_shot_map(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.get('x', []), df.get('y', []), alpha=0.6)
    ax.set_title('Shot map')
    return fig
