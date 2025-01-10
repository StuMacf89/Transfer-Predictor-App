# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:19:35 2025

@author: stuar
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import PyPizza, add_image, FontManager
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


page = st.sidebar.selectbox("Select a Page",['Transfer Performance Estimator','Customised Player Finder'])

if page == 'Transfer Performance Estimator': 
    # Load data
    df_tech = pd.read_csv('https://raw.githubusercontent.com/StuMacf89/Transfer-Predictor-App/refs/heads/main/StreamLit%20Dataset.csv')
    df_league_market_values = pd.read_csv('https://raw.githubusercontent.com/StuMacf89/Transfer-Predictor-App/refs/heads/main/StreamLit%20League%20Values.csv')
    df_club_market_values = pd.read_csv('https://raw.githubusercontent.com/StuMacf89/Transfer-Predictor-App/refs/heads/main/StreamLit%20Club%20Values%20Dataset.csv')
    df_club_market_values = df_club_market_values[df_club_market_values['Season'] == '2024/ 2025']
    df_league_market_values = df_league_market_values[df_league_market_values['Season'] == '2024/ 2025']
    
    df_tech = df_tech.rename(columns={'Position Specific Index': 'previous_club_performance'})
    
    df_tech = df_tech[df_tech['Market value'] != 0]
    scaler = StandardScaler()
    df_tech['previous_league_strength_scaled'] = scaler.fit_transform(df_tech[['previous_league_strength']])
    df_tech['previous_performance_relative_to_league'] = df_tech['previous_club_performance'] / df_tech['previous_league_strength_scaled']
    df_tech['previous_club_strength_scaled'] = scaler.fit_transform(df_tech[['previous_club_strength']])
    df_tech['previous_performance_relative_to_club'] = df_tech['previous_club_performance'] / df_tech['previous_club_strength_scaled']
    df_tech['market_value_scaled'] = scaler.fit_transform(df_tech[['Market value']])
    df_tech['value_minutes_interaction'] = df_tech['market_value_scaled'] * df_tech['Minutes played']
    df_tech['performance_per_euro'] = df_tech['previous_club_performance'] / df_tech['market_value_scaled']
    
    # Load the trained LightGBM model saved with joblib
    model = joblib.load("lgbm_classifier.pkl")
    
    # Streamlit App
    st.title("Transfer Peformance Estimator")
    
    
    # Sidebar for Position Selection
    st.sidebar.subheader("Select Position")
    positions = sorted(df_tech['main_position'].unique())
    selected_position = st.sidebar.selectbox("Choose a Position", positions)
    
    # Filter and Sort Players by Position and Performance
    filtered_players = df_tech[df_tech['main_position'] == selected_position]
    ranked_players = filtered_players.sort_values(by='previous_club_performance', ascending=False)
    
    # Add a slider to filter players by age
    min_age, max_age = int(ranked_players['Age'].min()), int(ranked_players['Age'].max())
    selected_age_range = st.sidebar.slider(
        "Filter players by age range:",
        min_age,
        max_age,
        (min_age, max_age),  # Default range is the min and max age
    )
    
    # Apply age filter
    ranked_players = ranked_players[(ranked_players['Age'] >= selected_age_range[0]) & (ranked_players['Age'] <= selected_age_range[1])]
    
    # Display Player Details
    st.write("Ranked List of Players Based On Performance:")
    columns_to_display = ['Player', 'main_position', 'previous_club_performance', 'Age', 'Comp_x', 'Team', 'Market value']
    st.dataframe(ranked_players[columns_to_display])
    
    # Sidebar for Player Selection with Typeahead Search
    st.sidebar.subheader("Select Player")
    
    # Use a selectbox directly for players, which supports typeahead natively
    selected_player = st.sidebar.selectbox(
        "Search or Choose a Player",
        ranked_players['Player'].tolist()  # The player list
    )
    
    # Radar Chart Function
    def cb_player_radar():
        params = [
            "Accurate\nPasses, %", "Accurate\nProgressive\nPasses, %", "Aerial\nDuels Won, %",
            "PAdj\nInterceptions", "Defensive\nDuels Won, %", "Progressive\nRuns per 90",
            "Successful\nDefensive\nActions per 90"
        ]
        
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        values = [
        round(player_data['Rank_Accurate passes, %'].values[0], 2),
        round(player_data['Rank_Accurate progressive passes, %'].values[0], 2),
        round(player_data['Rank_Aerial duels won, %'].values[0], 2),
        round(player_data['Rank_PAdj Interceptions'].values[0], 2),
        round(player_data['Rank_Defensive duels won, %'].values[0], 2),
        round(player_data['Rank_Progressive runs per 90'].values[0], 2),
        round(player_data['Rank_Successful defensive actions per 90'].values[0], 2),
        ]
    
        slice_colors = ["#cf9d1f"] * 3 + ["#13e83a"] * 2 + ["#0a69f0"] + ["#d60913"]
        text_colors = ["#000000"] * 7
        
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * 7, max_range=[1] * 7
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        return fig
    
    def fb_player_radar():
        params = ["Accurate\nProgressive\nPasses, %","Successful\nAttacking\nActions p90","Deep\Completions p90","xA",
            "Touches In\nBox p90", "Successful\nDefensive\nActions per 90",
            "Aerial\nDuels Won, %",
            "PAdj\nInterceptions", "Defensive\nDuels Won, %",
            
        ]
        
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        values = [
        round(player_data['Rank_Touches in box per 90'].values[0], 2),
        round(player_data['Rank_Successful attacking actions per 90'].values[0], 2),
        round(player_data['Rank_Aerial duels won, %'].values[0], 2),
        round(player_data['Rank_PAdj Interceptions'].values[0], 2),
        round(player_data['Rank_Defensive duels won, %'].values[0], 2),
        round(player_data['Rank_Accurate progressive passes, %'].values[0], 2),
        round(player_data['Rank_Successful defensive actions per 90'].values[0], 2),
        round(player_data['Rank_Deep completions per 90'].values[0], 2),
        round(player_data['Rank_xA'].values[0], 2),
        ]
    
        slice_colors = ["#cf9d1f"] * 1 + ["#13e83a"] * 4 + ["#0a69f0"]* 4 
        text_colors = ["#000000"] * 9
        
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * 9, max_range=[1] * 9
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        return fig
    
    def dm_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Passes\nper 90",
            "Accurate\nProgressive\nPasses, %",
            "Aerial\nDuels Won, %",
            "PAdj\nInterceptions",
            "Progressive\nRuns per 90",
            "Successful\nDefensive\nActions p90",
            "Successful\nAttacking\nActions p90",
            "xA",
            "Deep\nCompletions p90"
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        # Dynamically fetch the values for each feature
        features_dm_index = [
            'Rank_Passes per 90',
            'Rank_Accurate progressive passes, %',
            'Rank_Aerial duels won, %',
            'Rank_PAdj Interceptions',
            'Rank_Progressive runs per 90',
            'Rank_Successful defensive actions per 90',
            'Rank_Successful attacking actions per 90',
            'Rank_xA',
            'Rank_Deep completions per 90'
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_dm_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 2 + ["#13e83a"] * 4 + ["#0a69f0"] * 3
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        
        return fig
    
    def cm_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Passes\nper 90",
            "Accurate\nProgressive\nPasses, %",
            "Through\nPasses p90",
            "Successful\nDefensive\nActions p90",
            "Successful\nAttacking\nActions p90",
            "xA",
            "Deep\nCompletions p90",
            "Goals\nper 90",
            "Non-Penalty\nGoals",
            
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        # Dynamically fetch the values for each feature
        features_cm_index = [
            'Rank_Passes per 90',
            'Rank_Accurate progressive passes, %',
            'Rank_Through passes per 90',
            'Rank_Successful defensive actions per 90',
            'Rank_Successful attacking actions per 90',
            'Rank_xA',
            'Rank_Deep completions per 90',
            'Rank_Goals per 90',
            'Rank_Non-penalty goals',
            
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_cm_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 3 + ["#13e83a"] * 1 + ["#0a69f0"] * 5
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000") 
        
        return fig

    def am_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Passes\nper 90",
            "Accurate\nProgressive\nPasses, %",
            "Successful\nDefensive\nActions p90",
            "Successful\nAttacking\nActions p90",
            "xA",
            "Deep\nCompletions p90",
            "Goals\nper 90",
            "Non-Penalty\nGoals",
            "Key Passes\nper 90"
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        # Dynamically fetch the values for each feature
        features_am_index = [
            'Rank_Passes per 90',
            'Rank_Accurate progressive passes, %',
            'Rank_Successful defensive actions per 90',
            'Rank_Successful attacking actions per 90',
            'Rank_xA',
            'Rank_Deep completions per 90',
            'Rank_Goals per 90',
            'Rank_Non-penalty goals',
            'Rank_Key passes per 90'
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_am_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 2 + ["#13e83a"] * 1 + ["#0a69f0"] * 6
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953,"Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        
        return fig
    
    def wmf_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Key Passes\nper 90",
            "Accurate\nProgressive\nPasses, %",
            "Successful\nDefensive\nActions p90",
            "Successful\nAttacking\nActions p90",
            "xA",
            "Deep\nCompletions p90",
            "Goals\nper 90",
            "Non-Penalty\nGoals",
            "Accurate\nSmart Passes, %"
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        # Dynamically fetch the values for each feature
        features_wmf_index = [
            'Rank_Key passes per 90',
            'Rank_Accurate progressive passes, %',
            'Rank_Successful defensive actions per 90',
            'Rank_Successful attacking actions per 90',
            'Rank_xA',
            'Rank_Deep completions per 90',
            'Rank_Goals per 90',
            'Rank_Non-penalty goals',
            'Rank_Accurate smart passes, %'
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_wmf_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 1 + ["#13e83a"] * 4 + ["#0a69f0"] * 4
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        
        return fig

    def cf_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Non-Penalty\nGoals",
            "xG\nper 90",
            "Goals\nper 90"
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return

        # Dynamically fetch the values for each feature
        features_cf_index = [
            'Rank_Non-penalty goals',
            'Rank_xG per 90',
            'Rank_Goals per 90'
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_cf_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 1 + ["#13e83a"] * 1 + ["#0a69f0"] * 1
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        
        return fig
    
    def ss_player_radar():
        # Radar chart parameters (labels for each slice)
        params = [
            "Assists\nper 90",
            "xA\nper 90",
            "Non-Penalty\nGoals",
            "xG\nper 90",
            "Goals\nper 90",
            "Passes to\nPenalty Area\nper 90",
            "Successful\nAttacking\nActions per 90",
            "Key\nPasses\nper 90"
        ]
        
        # Ensure data for the selected player is available
        player_data = df_tech.loc[df_tech['Player'] == selected_player]
        if player_data.empty:
            st.error(f"No data available for the selected player: {selected_player}")
            return
    
        # Dynamically fetch the values for each feature
        features_ss_index = [
            'Rank_Assists per 90',
            'Rank_xA per 90',
            'Rank_Non-penalty goals',
            'Rank_xG per 90',
            'Rank_Goals per 90',
            'Rank_Passes to penalty area per 90',
            'Rank_Successful attacking actions per 90',
            'Rank_Key passes per 90'
        ]
        
        # Retrieve and round the feature values for the radar chart
        try:
            values = [round(player_data[feature].values[0], 2) for feature in features_ss_index]
        except KeyError as e:
            st.error(f"Missing feature in the dataset: {e}")
            return
    
        # Define slice and text colors for the chart
        slice_colors = ["#cf9d1f"] * 2 + ["#13e83a"] * 3 + ["#0a69f0"] * 3 
        text_colors = ["#000000"] * len(params)
        
        # Create the radar chart using PyPizza
        baker = PyPizza(
            params=params,
            background_color="#ffffff", straight_line_color="#EBEBE9", straight_line_lw=1,
            last_circle_lw=0, other_circle_lw=0, inner_circle_size=20,
            min_range=[0] * len(params), max_range=[1] * len(params)
        )
        fig, ax = baker.make_pizza(
            values, figsize=(8, 8.5), color_blank_space="same",
            slice_colors=slice_colors, value_colors=text_colors, value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(color="#000000", fontsize=11, va="center"),
            kwargs_values=dict(color="#000000", fontsize=11, zorder=3)
        )
        
        # Add title and subtitle
        fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
        fig.text(0.515, 0.953, "Percentile Rank vs Players in Same Position Within Respective League | Season 2023-24", size=13, ha="center", color="#000000")
        
        return fig
    
    # Display Radar Chart if Position is CB, LCB, or RCB
    if selected_position in ['CB', 'LCB', 'RCB']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = cb_player_radar()
            if fig:
                st.pyplot(fig)
    
    if selected_position in ['LB', 'RB', 'LWB','RWB']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = fb_player_radar()
            if fig:
                st.pyplot(fig)
                
    if selected_position in ['DMF', 'RDMF', 'LDMF']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = dm_player_radar()
            if fig:
                st.pyplot(fig)
    
    if selected_position in ['LCMF', 'RCMF']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = cm_player_radar()
            if fig:
                st.pyplot(fig)
                
    if selected_position in ['AMF', 'RAMF','LAMF']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = am_player_radar()
            if fig:
                st.pyplot(fig)
                
    if selected_position in ['RWF', 'LWF','RW','LW']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = wmf_player_radar()
            if fig:
                st.pyplot(fig)
    
    if selected_position in ['CF','SS']:
        if selected_player not in df_tech['Player'].values:
            st.error(f"No data available for the selected player: {selected_player}")
        else:
            st.write(f"Radar Chart for {selected_player}")
            fig = ss_player_radar()
            if fig:
                st.pyplot(fig)
    
    
    # Scatter Plot for Previous Club Performance vs Market Value
    st.subheader("Performance vs Market Value")
    
    # Calculate averages for the selected position
    average_performance = filtered_players['previous_club_performance'].mean()
    average_market_value = filtered_players['Market value'].mean()
    
    # Normalize the colors based on performance-to-value ratio
    norm = plt.Normalize(vmin=filtered_players['previous_club_performance'].min() / filtered_players['Market value'].max(),
                          vmax=filtered_players['previous_club_performance'].max() / filtered_players['Market value'].min())
    cmap = plt.cm.Blues  # Blue color gradient
    
    # Calculate the performance-to-value ratio for coloring
    filtered_players['performance_to_value_ratio'] = (
        filtered_players['previous_club_performance'] / filtered_players['Market value']
    )
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot for all players in the position
    scatter = ax.scatter(
        filtered_players['previous_club_performance'],
        filtered_players['Market value'],
        c=filtered_players['performance_to_value_ratio'],
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolor="k",
        alpha=0.9
    )
    
    # Highlight the selected player in red
    selected_player_data = filtered_players[filtered_players['Player'] == selected_player]
    ax.scatter(
        selected_player_data['previous_club_performance'],
        selected_player_data['Market value'],
        color="red",
        s=100,
        edgecolor="k",
        label=f"Selected Player: {selected_player}"
    )
    
    # Plot average lines
    ax.axhline(average_market_value, color='gray', linestyle='--', label='Average Market Value')
    ax.axvline(average_performance, color='gray', linestyle='--', label='Average Performance')
    
    # Add color bar for the gradient
    cbar = fig.colorbar(scatter, ax=ax, label="Performance-to-Value Ratio")
    
    # Set plot labels and title
    ax.set_title(f"Performance vs Market Value for {selected_position}", fontsize=14)
    ax.set_xlabel("Previous Club Performance", fontsize=12)
    ax.set_ylabel("Market Value (in Euros)", fontsize=12)
    
    # Move legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
    
    # Render the plot in Streamlit
    st.pyplot(fig)
    
    # Step 2: Select New League
    st.sidebar.subheader("Select New League")
    leagues = df_league_market_values['League'].unique()
    selected_league = st.sidebar.selectbox("Choose a New League", leagues)
    
    
    # Add 'new_league_strength' as a column to df_tech
    df_tech['new_league_strength'] = df_league_market_values[df_league_market_values['League'] == selected_league]['Value Aug 1 of Year'].mean()
    
    
    
    clubs = df_club_market_values[df_club_market_values['League'] == selected_league]
    clubs = clubs['Club'].unique()
    selected_club = st.sidebar.selectbox("Choose a New Club", clubs)
    
    # Extract new_club_strength
    new_club_strength = df_club_market_values[
        df_club_market_values['Club'] == selected_club
    ]['Value Aug 1 of Year'].mean()
    
    # Add 'new_club_strength' as a column to df_tech
    df_tech['new_club_strength'] = new_club_strength
    
    df_tech['club_strength_difference'] = df_tech['new_club_strength'] - df_tech['previous_club_strength']
    df_tech['league_strength_difference'] = df_tech['new_league_strength'] - df_tech['previous_club_strength']
    df_tech['performance_league_interaction'] = df_tech['previous_club_performance'] * df_tech['league_strength_difference']
    df_tech['performance_club_interaction'] = df_tech['previous_club_performance'] * df_tech['club_strength_difference']
    df_tech['new_league_strength_scaled'] = scaler.fit_transform(df_tech[['new_league_strength']])
    df_tech['new_club_strength_scaled'] = scaler.fit_transform(df_tech[['new_club_strength']])
    df_tech['club_strength_difference_scaled'] = df_tech['club_strength_difference']*2
    df_tech['previous_performance_league_interaction'] = df_tech['previous_club_performance'] * df_tech['previous_league_strength_scaled']
    df_tech['previous_performance_club_interaction'] = df_tech['previous_club_performance'] * df_tech['previous_club_strength_scaled']
    df_tech['previous_club_strength_relative_to_league'] = df_tech['previous_club_strength']* df_tech['previous_league_strength']
    df_tech['new_club_strength_relative_to_league'] = df_tech['new_club_strength']* df_tech['new_league_strength']
    df_tech['previous_performance_club_league_interaction'] = df_tech['previous_club_performance']*df_tech['previous_club_strength_relative_to_league']
    
    
    # Step 4: Build Model Inputs
    player_data = df_tech[df_tech['Player'] == selected_player].iloc[0]
    input_features = player_data[['value_minutes_interaction','Minutes played', 
                                    'Market value', 'performance_per_euro','Age','performance_league_interaction','performance_club_interaction',
                                    'previous_club_strength_relative_to_league','new_club_strength_relative_to_league']].values
    # Append new features
    model_inputs = np.append(input_features, [player_data['previous_performance_relative_to_league'], player_data['previous_performance_relative_to_club'],
                                              player_data['club_strength_difference'],player_data['league_strength_difference']]).reshape(1, -1)
    
    
    
    
    # Distribution Plot Section
    # Streamlit subheader for the section
    st.subheader("Position Specific Index Distribution in Previous League")
    
    # Filter data by the selected position (showing all players in the position)
    filtered_distribution_data = df_tech[df_tech['main_position'] == selected_position]
    
    # Ensure there is data to plot
    if filtered_distribution_data.empty:
        st.error(f"No data available for position: {selected_position}")
    else:
        # Remove NaN or null values from 'previous_club_performance'
        filtered_distribution_data = filtered_distribution_data.dropna(subset=['previous_club_performance'])
    
        # Get the selected player's performance
        highlight_data = filtered_distribution_data[filtered_distribution_data['Player'] == selected_player]
    
    if highlight_data.empty:
        st.error(f"No data available for selected player: {selected_player}")
    else:
        selected_performance = highlight_data['previous_club_performance'].values[0]

        # Create the histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalize the colors for the gradient effect
        norm = Normalize(vmin=filtered_distribution_data['previous_club_performance'].min(),
                         vmax=filtered_distribution_data['previous_club_performance'].max())
        cmap = plt.cm.RdYlGn  # Red-to-green color gradient

        # Create the histogram
        counts, bins, patches = ax.hist(
            filtered_distribution_data['previous_club_performance'],
            bins=30,
            color='grey',
            edgecolor='black'
        )

        # Apply the gradient colors to the histogram bars
        for bin_start, patch in zip(bins[:-1], patches):
            patch.set_facecolor(cmap(norm(bin_start)))

        # Highlight the selected player's performance
        ax.axvline(x=selected_performance, color='black', linestyle='--', lw=2)
        ax.text(
            selected_performance,
            max(counts) * 0.9,  # Place the label slightly below the top of the histogram
            f"{selected_player}",
            color='black',
            fontsize=12,
            ha='center'
        )

        # Set titles and labels
        ax.set_title(f"Histogram of previous season perforance for {selected_position} (All leagues)", fontsize=14)
        ax.set_xlabel("previous_club_performance", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)

        # Render the plot
        st.pyplot(fig)
    
    # Step 5: Make Predictions
    st.subheader("Prediction Results")
    st.markdown(
        "💡 **Note:** Consider the performance levels of the player when interpreting prediction, these predictions are based on historical performance and may vary depending on external factors and performance score (exceptionally high/ low)."
    )
    probability = model.predict_proba(model_inputs)[0][1]  # Use `predict_proba` to get probabilities
    decision = int(probability > 0.55)  # Decision threshold is 0.5
    
    # Display results
    st.write(f"**Probability of Player Achieving the Same Performance or Better in New League:** {probability:.2f}")
    st.write(f"**Predicted Decision:** {'Player is likely achieve the same performance or better from previous performance' if decision == 1 else 'Performance is not likely to improve from previous performance'}")
    
    
elif page == 'Customised Player Finder': 
        df_tech_radar = pd.read_csv("StreamLit Dataset.csv")
        df_tech_radar = df_tech_radar.rename(columns={
            col: col.replace('Rank_', '') for col in df_tech_radar.columns if col.startswith('Rank_')
        })
        
        # Set up Streamlit app
        st.title("Customised Player Finder")
        st.markdown(
            "💡 **Note:** There is no position selection option as your customisation may reveal players with the selected attributes but are playing in a different position"
        )
        
        st.sidebar.subheader("Select Features")
        
        # Define the columns of interest
        columns_of_interest = [
            'Aerial duels won, %', 'Accurate crosses, %', 'Defensive duels won, %', 'Accurate back passes, %', 
            'Accurate lateral passes, %', 'Accurate long passes, %', 'Aerial duels per 90', 'Successful dribbles, %',
            'Accurate passes, %', 'Assists', 'Assists per 90', 'Accurate forward passes, %', 
            'Accurate progressive passes, %', 'Average long pass length, m', 'Average pass length, m', 
            'Offensive duels won, %', 'Back passes per 90', 'Crosses per 90', 'Deep completed crosses per 90',
            'Accurate passes to final third, %', 'Deep completions per 90', 'Defensive duels per 90', 
            'Dribbles per 90', 'Forward passes per 90', 'Fouls per 90', 'Goal conversion, %', 'Goals', 
            'Goals per 90', 'Head goals', 'Head goals per 90', 'Accurate passes to penalty area, %', 
            'Interceptions per 90', 'Key passes per 90', 'Lateral passes per 90', 'Long passes per 90', 
            'Non-penalty goals', 'Non-penalty goals per 90', 'Accurate through passes, %', 'Offensive duels per 90', 
            'Shots on target, %', 'PAdj Interceptions', 'PAdj Sliding tackles', 'Passes per 90', 
            'Passes to final third per 90', 'Passes to penalty area per 90', 'Progressive passes per 90', 
            'Progressive runs per 90', 'Second assists per 90', 'Shots', 'Shots blocked per 90', 
            'Accurate smart passes, %', 'Shots per 90', 'Sliding tackles per 90', 'Smart passes per 90', 
            'Successful attacking actions per 90', 'Successful defensive actions per 90', 'Third assists per 90', 
            'Through passes per 90', 'Touches in box per 90', 'xA', 'xA per 90', 'xG', 'xG per 90', 
            'Shots per Goal', 'xgDifference'
        ]
        
        # Sidebar feature selection
        selected_features = st.sidebar.multiselect("Choose Features for Index Calculation", columns_of_interest)
        
        if selected_features:
            # Define the filtered dataframe based on selected features
            filtered_df = df_tech_radar[selected_features + ['Player', 'main_position','Age','Comp_x','Team']].dropna()
        
            # Define a function to calculate the boosted index
            def calculate_boosted_index(df, features):
                boosted_index = 0
                for feature in features:
                    # Calculate the 95th percentile for each feature
                    threshold = np.percentile(df[feature], 95)
                    # Logistic scaling for values above/below the threshold
                    boosted_feature = 1 / (1 + np.exp(-10 * (df[feature] - threshold) / threshold))
                    boosted_index += boosted_feature
                # Normalize the index
                boosted_index /= len(features)
                boosted_index = boosted_index / boosted_index.max()
                return boosted_index
        
            # Calculate the Position Specific Index
            filtered_df['Position Specific Index'] = calculate_boosted_index(filtered_df, selected_features)
        
            # Rank the results and display
            ranked_df = filtered_df.sort_values(by='Position Specific Index', ascending=False)
            st.write("Ranked Players Based on Position Specific Index:")
            columns_to_display = ['Player', 'main_position', 'Position Specific Index','Comp_x','Team','Age'] + selected_features
            st.dataframe(ranked_df[columns_to_display])
        
            # Radar Plot for a Specific Player
            st.sidebar.subheader("Select a Player for Radar Plot")
            selected_player = st.sidebar.selectbox("Choose a Player", ranked_df['Player'])
            
            def ss_player_radar():
                params = selected_features
                player_data = ranked_df[ranked_df['Player'] == selected_player]
        
                if player_data.empty:
                    st.error(f"No data available for the selected player: {selected_player}")
                    return None
        
                # Retrieve values for radar plot
                values = [round(player_data[feature].values[0], 2) for feature in params]
        
                # Define slice colors and text colors
                slice_colors = ["#cf9d1f"] * len(params)  # Customize as needed
                text_colors = ["#000000"] * len(params)
        
                # Create the radar chart
                baker = PyPizza(
                    params=params,
                    background_color="#ffffff",
                    straight_line_color="#EBEBE9",
                    straight_line_lw=1,
                    last_circle_lw=0,
                    other_circle_lw=0,
                    inner_circle_size=20,
                    min_range=[0] * len(params),
                    max_range=[1] * len(params),
                )
                fig, ax = baker.make_pizza(
                    values,
                    figsize=(8, 8.5),
                    color_blank_space="same",
                    slice_colors=slice_colors,
                    value_colors=text_colors,
                    value_bck_colors=slice_colors,
                    blank_alpha=0.4,
                    kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
                    kwargs_params=dict(color="#000000", fontsize=11, va="center"),
                    kwargs_values=dict(color="#000000", fontsize=11, zorder=3),
                )
        
                # Add title and subtitle
                fig.text(0.515, 0.975, selected_player, size=16, ha="center", color="#000000")
                fig.text(
                    0.515,
                    0.953,
                    "Percentile Rank vs Players Within Same Position and League | Season 2023-24",
                    size=13,
                    ha="center",
                    color="#000000",
                )
                return fig
        
            # Display the radar chart
            radar_chart = ss_player_radar()
            if radar_chart:
                st.pyplot(radar_chart)
            else:
                st.warning("Please select features to calculate the Position Specific Index.")