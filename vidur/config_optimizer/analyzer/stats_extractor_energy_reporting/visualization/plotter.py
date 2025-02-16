import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict
from .colors import COLORS

class EnergyVisualizer:
    def create_time_series_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create time series plot of energy consumption."""
        df = df.sort_values('time').dropna()
        
        # Downsampling
        n_points = 1000
        if len(df) > n_points:
            indices = np.linspace(0, len(df) - 1, n_points, dtype=int)
            df_downsampled = df.iloc[indices]
        else:
            df_downsampled = df
            
        fig = go.Figure()
        
        # Add energy consumption line
        fig.add_trace(go.Scatter(
            x=df_downsampled["time"].tolist(),
            y=df_downsampled["energy"].tolist(),
            name="Energy Consumption",
            mode="lines",
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate="Time: %{x:.2f}s<br>Energy: %{y:.4f} kWh<extra></extra>"
        ))
        
        # Add MFU on secondary axis - values are already percentages
        fig.add_trace(go.Scatter(
            x=df_downsampled["time"].tolist(),
            y=df_downsampled["mfu"].tolist(),  # No multiplication needed
            name="Model FLOPs Utilization",
            mode="lines",
            line=dict(color=COLORS['secondary'], width=2),
            yaxis="y2",
            hovertemplate="Time: %{x:.2f}s<br>MFU: %{y:.1f}%<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Energy Consumption and Model Utilization Over Time",
                font=dict(color=COLORS['text'])
            ),
            xaxis=dict(
                title=dict(
                    text="Time (seconds)",
                    font=dict(color=COLORS['text'])
                ),
                gridcolor=COLORS['green_palette'][0],
                zeroline=False
            ),
            yaxis=dict(
                title=dict(
                    text="Energy Consumption (kWh)",
                    font=dict(color=COLORS['text'])
                ),
                gridcolor=COLORS['green_palette'][0],
                zeroline=False
            ),
            yaxis2=dict(
                title=dict(
                    text="Model FLOPs Utilization (%)",
                    font=dict(color=COLORS['text'])
                ),
                overlaying="y",
                side="right",
                gridcolor=COLORS['green_palette'][0],
                zeroline=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def create_efficiency_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of power usage vs MFU."""
        df = df.dropna()
        
        # Downsample if needed
        n_points = 500
        if len(df) > n_points:
            indices = np.linspace(0, len(df) - 1, n_points, dtype=int)
            df_downsampled = df.iloc[indices]
        else:
            df_downsampled = df
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_downsampled["mfu"].tolist(),
            y=df_downsampled["effective_power"].tolist(),
            mode="markers",
            marker=dict(
                size=8,
                color=df_downsampled["effective_power"].tolist(),
                colorscale=[
                    [0, COLORS['green_palette'][0]],    # Lightest green
                    [0.25, COLORS['green_palette'][1]], 
                    [0.5, COLORS['green_palette'][2]],
                    [0.75, COLORS['green_palette'][3]],
                    [1, COLORS['green_palette'][4]]     # Darkest green
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Power Usage (W)",
                        font=dict(color=COLORS['text'])
                    )
                )
            ),
            hovertemplate=(
                "MFU: %{x:.1f}%<br>" +
                "Power: %{y:.1f}W<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title="Power Usage vs Model Utilization",
            xaxis_title="Model FLOPs Utilization (%)",
            yaxis_title="Power Usage (W)",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        return fig

    def create_regional_impact_plot(self, metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create regional impact comparison plot."""
        regions = list(metrics.keys())
        carbon_emissions = [metrics[r]['carbon_emissions'] for r in regions]
        energy_costs = [metrics[r]['energy_cost'] for r in regions]
        
        fig = go.Figure()
        
        # Add carbon emissions bar
        fig.add_trace(go.Bar(
            y=regions,
            x=carbon_emissions,
            name="Carbon Emissions (gCO2eq)",
            orientation='h',
            marker=dict(color=COLORS['primary'])
        ))
        
        # Add energy cost bar
        fig.add_trace(go.Bar(
            y=regions,
            x=energy_costs,
            name="Energy Cost ($)",
            orientation='h',
            marker=dict(color=COLORS['secondary'])
        ))
        
        fig.update_layout(
            title=dict(
                text="Regional Impact Comparison",
                font=dict(color=COLORS['text'])
            ),
            xaxis=dict(
                title=dict(
                    text="Impact Metrics",
                    font=dict(color=COLORS['text'])
                )
            ),
            yaxis=dict(
                title=dict(
                    text="Region",
                    font=dict(color=COLORS['text'])
                )
            ),
            barmode='group',
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig 