from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_truss_data_for_iteration(TRUSS1, X_init_single, point_index):
    current_item = X_init_single[point_index]
    TRUSS1.Write_call_state(current_item.numpy(), final=False)
    return TRUSS1.state['nodes'], TRUSS1.state['member_df']

def loss_function_figure(total_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(total_loss)),  # Assuming each row corresponds to an iteration
        y=total_loss,
        mode='lines',
        name='Total Loss',
        line=dict(color='#00A6D6')  # Fixed opacity
    ))

    min_loss_index = np.argmin(total_loss)
    fig.add_trace(go.Scatter(
        x=[min_loss_index],
        y=[total_loss[min_loss_index]],
        mode='markers',
        name='Minimum Loss',
        marker=dict(color='#EC6842', size=6)
    ))

    min_mass_index = 30
    fig.add_trace(go.Scatter(
        x=[min_mass_index],
        y=[total_loss[min_mass_index]],
        mode='markers',
        name='Best Solution',
        marker=dict(color='#E03C31', size=8, symbol='star')
    ))

    fig.update_layout(
        title={
            'text': "Loss per Iteration"
        },
        xaxis=dict(
            title="Iteration",
            showline=True,
            showgrid=True,
            ticks='outside'),
        yaxis=dict(
            title="Total Loss",
            showline=True,
            showgrid=True,
            ticks='outside'
        ),
        width=650,
        height=500,
        legend_title_text='Metric',
        legend_font_size=12,
        margin=dict(l=100, r=100, t=100, b=100),
        template = "plotly_white"
    )
    return fig

def plot_truss_layout(nodes, member_df):
    fig = go.Figure()    
    for _, row in member_df.iterrows():
        node1 = row['Node #1']
        node2 = row['Node #2']
        x0, y0 = nodes[node1]
        x1, y1 = nodes[node2]
       
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',line=dict(color='#0C2340'),
                                 name=f'Element between {node1} - {node2}'))

    node_x = [coord[0] for coord in nodes.values()]
    node_y = [coord[1] for coord in nodes.values()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', 
                             marker=dict(size=5, color='black', symbol='circle'),
                             name='Nodes'))

    fig.add_trace(go.Scatter(x=[node_x[0], node_x[-1]], y=[node_y[0]-0.02, node_y[-1]-0.02], mode='markers',
                         marker=dict(size=15, color='#00B8C8', symbol='triangle-up'),
                         name='Supports'))

    fig.update_layout(
        title={
            'text': 'Truss Layout at given timestep'
        },
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        showlegend=False,
        xaxis=dict(
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True,
        ),
        margin=dict(l=100, r=100, t=100, b=100),
        height=500,
        width=900,
        template = "plotly_white"
    )
    return fig