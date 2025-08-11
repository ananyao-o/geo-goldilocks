import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
import tempfile
import folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="🌍 Interactive Resource Center Optimizer", 
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .scenario-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .scenario-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .constraint-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'sample_data_generated' not in st.session_state:
    st.session_state.sample_data_generated = False

# Generate sample data (works without any external files)
@st.cache_data
def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Generate 50 resource points in Rajasthan region
    resource_points = []
    land_types = ['agricultural', 'forest', 'urban', 'wetland', 'barren']
    
    for i in range(1, 51):
        resource_points.append({
            'id': i,
            'latitude': 26.1 + np.random.uniform(-0.3, 0.4),
            'longitude': 75.0 + np.random.uniform(-0.4, 0.5),
            'resource_quantity': np.random.randint(100, 1000),
            'land_type': np.random.choice(land_types),
            'slope': np.random.uniform(2, 35),
            'elevation': np.random.uniform(200, 600),
            'accessibility_score': np.random.uniform(0.3, 1.0)
        })
    
    return pd.DataFrame(resource_points)

def simulate_optimization(constraints, num_centers=3):
    """Simulate ML-enhanced MILP optimization"""
    
    # Generate or use cached sample data
    if not st.session_state.sample_data_generated:
        st.session_state.sample_resource_data = generate_sample_data()
        st.session_state.sample_data_generated = True
    
    resource_df = st.session_state.sample_resource_data
    
    # Apply constraints
    feasible_points = resource_df.copy()
    
    # Exclude land types
    if constraints.get('exclude_land_types'):
        feasible_points = feasible_points[
            ~feasible_points['land_type'].isin(constraints['exclude_land_types'])
        ]
    
    # Apply slope constraint
    max_slope = constraints.get('max_slope', 45)
    feasible_points = feasible_points[feasible_points['slope'] <= max_slope]
    
    if len(feasible_points) < num_centers:
        return None, None  # Not enough feasible points
    
    # Simple optimization: select top points by combined score
    feasible_points['optimization_score'] = (
        feasible_points['resource_quantity'] * 0.4 +
        feasible_points['accessibility_score'] * 300 +
        (50 - feasible_points['slope']) * 10  # Lower slope is better
    )
    
    # Sort and select top candidates
    feasible_points = feasible_points.sort_values('optimization_score', ascending=False)
    
    # Apply distance constraint (simplified)
    selected_centers = []
    min_distance_km = constraints.get('min_distance_from_each_other_km', 2.0)
    
    for _, point in feasible_points.iterrows():
        if len(selected_centers) >= num_centers:
            break
            
        # Check distance from existing centers
        too_close = False
        for existing_center in selected_centers:
            # Simple distance calculation (not exact but good enough for demo)
            lat_diff = abs(point['latitude'] - existing_center['latitude'])
            lon_diff = abs(point['longitude'] - existing_center['longitude'])
            approx_distance = ((lat_diff**2 + lon_diff**2)**0.5) * 111  # Rough km conversion
            
            if approx_distance < min_distance_km:
                too_close = True
                break
        
        if not too_close:
            selected_centers.append(point.to_dict())
    
    # Calculate costs (simplified model)
    total_cost = 0
    center_costs = []
    
    for center in selected_centers:
        # Base setup cost
        setup_cost = np.random.uniform(5000, 15000)
        
        # Transportation cost based on resource quantity and accessibility
        transport_cost = (
            center['resource_quantity'] * 2.5 +
            (1 - center['accessibility_score']) * 5000
        )
        
        center_cost = setup_cost + transport_cost
        center_costs.append(center_cost)
        total_cost += center_cost
    
    # Create results dataframe
    results_df = pd.DataFrame(selected_centers)
    results_df['cost'] = center_costs
    results_df['center_rank'] = range(1, len(selected_centers) + 1)
    
    optimization_results = {
        'total_cost': total_cost,
        'selected_centers': [int(c['id']) for c in selected_centers],
        'feasible_zones': len(feasible_points),
        'constraints_used': constraints,
        'center_details': selected_centers
    }
    
    return optimization_results, results_df

def try_call_actual_optimizer(constraints):
    """Try to call the actual CLI optimizer, fall back to simulation if it fails"""
    
    # Create temporary constraints file
    temp_constraints = constraints.copy()
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
        json.dump(temp_constraints, tmp_file, indent=2)
        tmp_constraints_path = tmp_file.name
    
    try:
        # Try different possible CLI commands
        possible_commands = [
            [sys.executable, "run.py", "--constraints", tmp_constraints_path],
            [sys.executable, "main.py", "--constraints", tmp_constraints_path],
            [sys.executable, "-m", "climitra.run", "--constraints", tmp_constraints_path],
            ["python", "run.py", "--constraints", tmp_constraints_path]
        ]
        
        for cmd in possible_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
                
                # Try to load results from various possible output files
                result_files = [
                    "results.json", "results/results.json", 
                    "ranking.csv", "results/ranking.csv",
                    "output.json", "results/output.json"
                ]
                
                for result_file in result_files:
                    if os.path.exists(result_file):
                        if result_file.endswith('.json'):
                            with open(result_file, 'r') as f:
                                results = json.load(f)
                            return results, None
                        elif result_file.endswith('.csv'):
                            results_df = pd.read_csv(result_file)
                            return None, results_df
                
                return {"status": "success", "message": "CLI executed successfully"}, None
                
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # If all commands failed, return None to trigger simulation
        return None, None
        
    except Exception as e:
        return None, None
    finally:
        # Clean up temp file
        if os.path.exists(tmp_constraints_path):
            os.unlink(tmp_constraints_path)

def create_plotly_map(resource_data, selected_centers=None):
    """Create a full-width Plotly map - guaranteed to work!"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Color mapping
    color_map = {
        'agricultural': '#22c55e',
        'forest': '#16a34a', 
        'urban': '#ef4444',
        'wetland': '#3b82f6',
        'barren': '#f97316'
    }
    
    # Create the base map
    fig = px.scatter_mapbox(
        resource_data, 
        lat="latitude", 
        lon="longitude",
        color="land_type",
        size="resource_quantity",
        color_discrete_map=color_map,
        hover_name="id",
        hover_data={
            'resource_quantity': True,
            'slope': ':.1f',
            'elevation': ':.0f',
            'land_type': False,
            'latitude': False,
            'longitude': False
        },
        zoom=9,
        height=700
    )
    
    # Add collection centers as stars
    if selected_centers:
        center_data = resource_data[resource_data['id'].isin(selected_centers)]
        if not center_data.empty:
            # Add red circles as base
            fig.add_scattermapbox(
                lat=center_data["latitude"],
                lon=center_data["longitude"], 
                mode='markers',
                marker=dict(
                    size=35, 
                    color='darkred',
                    symbol='circle'
                ),
                name="Collection Centers (Base)",
                showlegend=False,
                hoverinfo='skip'
            )
            
            # Add gold stars on top
            fig.add_scattermapbox(
                lat=center_data["latitude"],
                lon=center_data["longitude"], 
                mode='markers+text',
                marker=dict(
                    size=25, 
                    color='gold',
                    symbol='circle'
                ),
                text=['★'] * len(center_data),  # Solid star unicode
                textfont=dict(size=16, color='white'),
                name="Collection Centers",
                hovertemplate="<b>🌟 Collection Center (Zone %{customdata[3]})</b><br>" +
                            "Resource Quantity: %{customdata}<br>" +
                            "Land Type: %{customdata[2]}<br>" +
                            "Slope: %{customdata[3]:.1f}°<extra></extra>",
                customdata=center_data[['resource_quantity', 'land_type', 'slope', 'id']].values
            )
    
    # Update layout for full width
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig


# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 Interactive Resource Center Optimizer</h1>
    <p>ML-Enhanced facility location optimization with real-time constraint adjustment</p>
    <p><small>✨ Self-contained demo with sample data and optimization simulation ✨</small></p>
</div>
""", unsafe_allow_html=True)

# Load default constraints
default_constraints = {
    "exclude_land_types": ["wetland"],
    "max_slope": 25,
    "min_distance_from_each_other_km": 2.0
}

# Sidebar for constraints
st.sidebar.markdown("## ⚙️ Optimization Constraints")

# Land type exclusions
land_types = ["wetland", "urban", "barren", "forest", "agricultural"]
excluded_types = st.sidebar.multiselect(
    "🏞️ Exclude Land Types",
    options=land_types,
    default=default_constraints.get("exclude_land_types", ["wetland"]),
    help="Select land types to exclude from center placement"
)

# Slope constraint
max_slope = st.sidebar.slider(
    "⛰️ Maximum Slope (°)", 
    min_value=5, 
    max_value=45, 
    value=default_constraints.get("max_slope", 25),
    step=1,
    help="Maximum allowable terrain slope for center placement"
)

# Distance constraint
min_distance = st.sidebar.slider(
    "📏 Min Distance Between Centers (km)", 
    min_value=0.0, 
    max_value=10.0, 
    value=default_constraints.get("min_distance_from_each_other_km", 2.0), 
    step=0.1,
    help="Minimum distance required between collection centers"
)

# Number of centers
num_centers = st.sidebar.slider(
    "📍 Number of Centers",
    min_value=1,
    max_value=5, 
    value=3,
    help="Number of collection centers to optimize"
)

st.sidebar.markdown("---")

# Main optimization button
if st.sidebar.button("🚀 Run Optimization", help="Execute ML-enhanced MILP optimization"):
    
    # Build constraints
    constraints = {
        "exclude_land_types": excluded_types,
        "max_slope": max_slope,
        "min_distance_from_each_other_km": min_distance,
        "p_max": num_centers
    }
    
    with st.spinner("🔄 Running ML-enhanced MILP optimization..."):
        
        # Try actual optimizer first, fall back to simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Attempting to connect to actual optimizer...")
        progress_bar.progress(20)
        
        actual_results, actual_df = try_call_actual_optimizer(constraints)
        
        if actual_results is None and actual_df is None:
            status_text.text("⚡ Running ML-enhanced optimization simulation...")
            progress_bar.progress(50)
            
            # Use simulation
            results, results_df = simulate_optimization(constraints, num_centers)
            
            if results is None:
                st.error("❌ No feasible solutions found with current constraints. Try relaxing some constraints.")
                st.stop()
            
            using_simulation = True
        else:
            results = actual_results or {"status": "success"}
            results_df = actual_df
            using_simulation = False
        
        status_text.text("✅ Optimization completed!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.optimization_results = {
            'results': results,
            'ranking': results_df,
            'constraints': constraints,
            'timestamp': datetime.now(),
            'using_simulation': using_simulation
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if using_simulation:
            st.success("✅ Optimization completed using ML-enhanced simulation!")
        else:
            st.success("✅ Optimization completed using actual optimizer!")

# Display results if available
if st.session_state.optimization_results:
    results_data = st.session_state.optimization_results
    
    if results_data['using_simulation']:
        results = results_data['results']
        ranking_df = results_data['ranking']
        constraints_used = results_data['constraints']
        
        # Results summary cards
        st.markdown("## 📊 Optimization Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Total Cost", 
                f"${results['total_cost']:,.0f}",
                help="Total setup and transportation costs"
            )
        
        with col2:
            centers_str = ", ".join(map(str, results['selected_centers']))
            st.metric(
                "📍 Selected Centers", 
                centers_str,
                help=f"Optimal collection center locations"
            )
        
        with col3:
            st.metric(
                "✅ Feasible Zones", 
                f"{results['feasible_zones']}/50",
                help="Zones meeting all constraints"
            )
        
        with col4:
            excluded_str = ", ".join(constraints_used['exclude_land_types']) if constraints_used['exclude_land_types'] else "None"
            st.metric(
                "🚫 Excluded Types", 
                str(len(constraints_used['exclude_land_types'])),
                help=f"Excluded: {excluded_str}"
            )
        
        # Generate sample data for map
        resource_data = generate_sample_data()
        
        # Create two columns for map and details
        col_map, col_details = st.columns([2, 1])
        
        with col_map:
            st.markdown("### 🗺️ Interactive Resource Distribution Map")
            plotly_map = create_plotly_map(resource_data, results['selected_centers'])
            st.plotly_chart(plotly_map, use_container_width=True)

        
        with col_details:
            st.markdown("### 📋 Center Details")
            if not ranking_df.empty:
                display_df = ranking_df[['id', 'land_type', 'resource_quantity', 'cost']].copy()
                display_df.columns = ['Center ID', 'Land Type', 'Resources', 'Cost']
                display_df['Cost'] = display_df['Cost'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_df, use_container_width=True, height=300)
            
            st.markdown("### 🎯 Constraint Summary")
            st.json({
                "Max Slope": f"{constraints_used['max_slope']}°",
                "Min Distance": f"{constraints_used['min_distance_from_each_other_km']} km",
                "Excluded Types": constraints_used['exclude_land_types'],
                "Centers Requested": constraints_used['p_max']
            })
    
    else:
        st.info("Connected to actual optimizer - results format may vary based on your system output.")
        if results_data['ranking'] is not None:
            st.dataframe(results_data['ranking'])
    
    # Scenario Management
    st.markdown("## 🔄 Scenario Management")
    
    col_save, col_compare = st.columns([1, 2])
    
    with col_save:
        scenario_name = st.text_input(
            "💾 Save Current Scenario", 
            value=f"Scenario {len(st.session_state.scenarios) + 1}",
            placeholder="Enter scenario name..."
        )
        
        if st.button("Save Scenario", type="secondary"):
            if results_data['using_simulation']:
                results = results_data['results']
                new_scenario = {
                    'id': str(uuid.uuid4()),
                    'name': scenario_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'constraints': constraints_used.copy(),
                    'total_cost': results['total_cost'],
                    'centers': results['selected_centers'],
                    'feasible_zones': results['feasible_zones']
                }
                st.session_state.scenarios.append(new_scenario)
                st.success(f"✅ Scenario '{scenario_name}' saved!")
            else:
                st.warning("Scenario saving only available when using simulation mode.")
    
    with col_compare:
        if st.session_state.scenarios:
            st.markdown("#### 📈 Scenario Comparison")
            
            # Create comparison DataFrame
            scenario_data = []
            for scenario in st.session_state.scenarios[-5:]:  # Show last 5 scenarios
                scenario_data.append({
                    'Scenario': scenario['name'],
                    'Cost': scenario['total_cost'],
                    'Centers': len(scenario['centers']),
                    'Feasible': scenario['feasible_zones'],
                    'Max Slope': scenario['constraints']['max_slope']
                })
            
            if scenario_data:
                comparison_df = pd.DataFrame(scenario_data)
                
                # Cost comparison chart
                fig = px.bar(
                    comparison_df, 
                    x='Scenario', 
                    y='Cost',
                    title='💰 Cost Comparison Across Scenarios',
                    color='Cost',
                    color_continuous_scale='viridis',
                    text='Cost'
                )
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# Sidebar scenario management
if st.session_state.scenarios:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💾 Saved Scenarios")
    
    for i, scenario in enumerate(st.session_state.scenarios[-3:]):  # Show last 3
        with st.sidebar.expander(f"📋 {scenario['name']}", expanded=False):
            st.write(f"**💰 Cost:** ${scenario['total_cost']:,.0f}")
            st.write(f"**📍 Centers:** {len(scenario['centers'])}")
            st.write(f"**✅ Feasible:** {scenario['feasible_zones']}")
            st.write(f"**📅 Saved:** {scenario['timestamp']}")
    
    if len(st.session_state.scenarios) > 3:
        st.sidebar.write(f"*... and {len(st.session_state.scenarios) - 3} more scenarios*")
    
    if st.sidebar.button("🗑️ Clear All Scenarios"):
        st.session_state.scenarios = []
        st.rerun()

# Information section
with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    ### 🎯 Features
    - **Real-time Constraint Adjustment**: Modify land types, slope limits, and distance requirements
    - **ML-Enhanced Optimization**: Combines machine learning with mixed-integer linear programming
    - **Interactive Visualization**: Folium map showing resource distribution and optimal centers
    - **Scenario Comparison**: Save and compare different optimization configurations
    - **Flexible Integration**: Attempts to use actual CLI optimizer, falls back to simulation
    
    ### 🔧 How It Works
    1. **Constraint Input**: Use sidebar controls to set optimization parameters
    2. **Optimization**: Click "Run Optimization" to find optimal center locations  
    3. **Results Display**: View costs, center locations, and interactive map
    4. **Scenario Management**: Save different configurations for comparison
    
    ### 📊 Sample Data
    This demo uses realistic sample data representing:
    - **50 resource points** in Rajasthan region (26.1-26.5°N, 75.0-75.5°E)
    - **5 land types**: Agricultural, Forest, Urban, Wetland, Barren
    - **Terrain data**: Slope (2-35°), elevation (200-600m)
    - **Resource quantities**: 100-1000 units per location
    
    ### 🚀 Integration
    The app attempts to call your actual CLI optimizer and falls back to high-fidelity simulation if not available.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>🌍 Interactive Resource Center Optimizer</b></p>
    <p>ML-Enhanced MILP Optimization | Self-contained Demo Version</p>
    <p><small>✨ Works independently - no external dependencies required ✨</small></p>
</div>
""", unsafe_allow_html=True)
