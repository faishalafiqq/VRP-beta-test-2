import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import io
import json

st.set_page_config(
    page_title="🚛 VRP Banjir Live Pro",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# POPPINS FONT + MODERN UI
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
* {
    font-family: 'Poppins', sans-serif !important;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}

.main {
    background: #0f172a;
}

/* Header */
.main-header {
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 2rem !important;
    letter-spacing: -1px;
    text-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
}

.subheader {
    font-size: 0.95rem;
    color: #94a3b8;
    text-align: center;
    margin-top: -1.5rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.8rem;
    border-radius: 16px;
    color: #e2e8f0;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3), 0 0 20px rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.2);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(10px);
}

.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 50px rgba(59, 130, 246, 0.2), 0 0 30px rgba(139, 92, 246, 0.15);
    border-color: rgba(148, 163, 184, 0.4);
}

.best-method {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 0 15px 50px rgba(16, 185, 129, 0.3) !important;
    border-color: rgba(16, 185, 129, 0.5) !important;
}

.best-method:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 60px rgba(16, 185, 129, 0.4) !important;
}

/* Input Fields */
.stNumberInput, .stSlider, .stTextInput input, .stSelectbox select {
    background: #1e293b !important;
    border: 2px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500;
    transition: all 0.3s ease !important;
}

.stNumberInput:focus, .stTextInput input:focus, .stSelectbox select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    font-family: 'Poppins', sans-serif !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(59, 130, 246, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #1e293b;
    border-radius: 10px;
    color: #94a3b8;
    border: 2px solid #334155;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: white;
    border-color: #3b82f6;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
}

/* DataFrames */
.stDataFrame {
    background: #1e293b !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    border: 1px solid #334155 !important;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

/* Alerts */
.stAlert {
    background: #1e293b;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    color: #e2e8f0;
    font-weight: 500;
}

.stSuccess {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-left-color: #34d399 !important;
}

.stWarning {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    border-left-color: #fbbf24 !important;
}

.stError {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    border-left-color: #f87171 !important;
}

.stInfo {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    border-left-color: #60a5fa !important;
}

/* Section Headers */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

h1 {
    color: #f1f5f9 !important;
    font-size: 2rem !important;
}

h2 {
    color: #e2e8f0 !important;
    font-size: 1.5rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}

h3 {
    color: #cbd5e1 !important;
    font-size: 1.1rem !important;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 2rem 0;
}

/* Badge-like styling */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

/* Radio buttons */
.stRadio > label {
    background: #1e293b;
    padding: 10px 16px;
    border-radius: 10px;
    border: 2px solid #334155;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
}

.stRadio > label:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

/* Expander */
.stExpander {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* Metric values */
.metric-value {
    font-weight: 700;
    font-size: 1.8rem;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Spinner */
.stSpinner > div {
    border-color: #3b82f6 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
""", unsafe_allow_html=True)

TOMTOM_API_KEY = "DPKi6pXg3JG1rT3aKI8t7PWCzjYcxIof"

FLOOD_THRESHOLDS = {'high': {'precip': 20, 'prob': 70}, 'medium': {'precip': 10, 'prob': 60}, 'low': {'precip': 5, 'prob': 40}}

# 6 VRP ALGORITHMS
class VRP_MasterSolver:
    def __init__(self, dist_matrix, demands, capacity):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.nodes = list(range(1, len(dist_matrix)))
    
    def solve_nn(self):
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            curr = 0
            route = []
            load = 0
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                next_node = min(cands, key=lambda x: self.dist_matrix[curr][x])
                route.append(next_node)
                unvisited.remove(next_node)
                load += self.demands[next_node]
                curr = next_node
            if route: routes.append(route)
        return routes
    
    def solve_cw(self):
        routes = [[i] for i in self.nodes]
        savings = []
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    s = self.dist_matrix[i][0] + self.dist_matrix[0][j] - self.dist_matrix[i][j]
                    savings.append((s, i, j))
        savings.sort(key=lambda x: x[0], reverse=True)
        for s, i, j in savings:
            ri = next((idx for idx, r in enumerate(routes) if i in r), None)
            rj = next((idx for idx, r in enumerate(routes) if j in r), None)
            if ri is not None and rj is not None and ri != rj:
                r1, r2 = routes[ri], routes[rj]
                if sum(self.demands[n] for n in r1 + r2) <= self.capacity:
                    if r1[-1] == i and r2[0] == j:
                        routes[ri].extend(r2)
                        routes.pop(rj)
                    elif r2[-1] == j and r1[0] == i:
                        routes[rj].extend(r1)
                        routes.pop(ri)
        return [r for r in routes if r]
    
    def solve_cheapest_insertion(self):
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            seed = min(unvisited, key=lambda x: self.dist_matrix[0][x])
            route = [seed]
            unvisited.remove(seed)
            load = self.demands[seed]
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                best_cost = float('inf')
                best_n, best_p = None, None
                full_r = [0] + route + [0]
                for n in cands:
                    for i in range(len(full_r) - 1):
                        u, v = full_r[i], full_r[i + 1]
                        cost_add = self.dist_matrix[u][n] + self.dist_matrix[n][v] - self.dist_matrix[u][v]
                        if cost_add < best_cost:
                            best_cost, best_n, best_p = cost_add, n, i
                if best_n:
                    route.insert(best_p, best_n)
                    unvisited.remove(best_n)
                    load += self.demands[best_n]
                else: break
            routes.append(route)
        return routes
    
    def solve_nearest_insertion(self):
        return self._solve_insertion_general('nearest')
    
    def solve_farthest_insertion(self):
        return self._solve_insertion_general('farthest')
    
    def solve_arbitrary_insertion(self):
        return self._solve_insertion_general('arbitrary')
    
    def _solve_insertion_general(self, mode):
        unvisited = set(self.nodes)
        routes = []
        while unvisited:
            if mode == 'farthest':
                seed = max(unvisited, key=lambda x: self.dist_matrix[0][x])
            elif mode == 'arbitrary':
                seed = list(unvisited)[0]
            else:
                seed = min(unvisited, key=lambda x: self.dist_matrix[0][x])
            route = [seed]
            unvisited.remove(seed)
            load = self.demands[seed]
            while True:
                cands = [n for n in unvisited if load + self.demands[n] <= self.capacity]
                if not cands: break
                if mode == 'nearest':
                    sel_node = min(cands, key=lambda c: min([self.dist_matrix[c][r] for r in route] + [self.dist_matrix[c][0]]))
                    check_list = [sel_node]
                else:
                    check_list = cands[:3]
                best_cost = float('inf')
                best_node, best_pos = None, None
                full_r = [0] + route + [0]
                for n in check_list:
                    for i in range(len(full_r) - 1):
                        u, v = full_r[i], full_r[i + 1]
                        cost_add = self.dist_matrix[u][n] + self.dist_matrix[n][v] - self.dist_matrix[u][v]
                        if cost_add < best_cost:
                            best_cost, best_node, best_pos = cost_add, n, i
                if best_node:
                    route.insert(best_pos, best_node)
                    unvisited.remove(best_node)
                    load += self.demands[best_node]
                else: break
            routes.append(route)
        return routes

def get_weather_forecast(locations_df):
    weather_data = {}
    base_url = "https://api.open-meteo.com/v1/forecast"
    progress_bar = st.progress(0)
    for idx, row in locations_df.iterrows():
        progress_bar.progress((idx + 1) / len(locations_df))
        params = {'latitude': row['Latitude'], 'longitude': row['Longitude'], 'hourly': 'precipitation,precipitation_probability', 'forecast_days': 1, 'timezone': 'Asia/Jakarta'}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            hours = 6
            precip = data['hourly']['precipitation'][:hours]
            prob = data['hourly']['precipitation_probability'][:hours]
            avg_precip = np.mean(precip) if precip else 0
            max_prob = max(prob) if prob else 0
            if avg_precip > FLOOD_THRESHOLDS['high']['precip'] or max_prob > FLOOD_THRESHOLDS['high']['prob']:
                level = "🔴 BANJIR BESAR"
            elif avg_precip > FLOOD_THRESHOLDS['medium']['precip'] or max_prob > FLOOD_THRESHOLDS['medium']['prob']:
                level = "🟠 GENANGAN"
            elif avg_precip > FLOOD_THRESHOLDS['low']['precip'] or max_prob > FLOOD_THRESHOLDS['low']['prob']:
                level = "🟡 HUJAN LEBAT"
            else:
                level = "🟢 AMAN"
            weather_data[idx] = {'avg_precip': round(avg_precip, 1), 'max_prob': int(max_prob), 'flood_level': level}
        except:
            weather_data[idx] = {'avg_precip': 0, 'max_prob': 0, 'flood_level': "❓ ERROR"}
    progress_bar.empty()
    return weather_data

def get_distance_matrix(locations_df):
    n = len(locations_df)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
                continue
            lat1, lon1 = locations_df.iloc[i][['Latitude', 'Longitude']].values
            lat2, lon2 = locations_df.iloc[j][['Latitude', 'Longitude']].values
            R = 6371000
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi, dlambda = np.radians(lat2-lat1), np.radians(lon2-lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist_matrix[i][j] = R * c
    return dist_matrix

def get_real_route_osm(start_lat, start_lon, end_lat, end_lon):
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {'overview': 'full', 'geometries': 'geojson'}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if 'routes' in data and len(data['routes']) > 0:
            coords = data['routes'][0]['geometry']['coordinates']
            return [[c[1], c[0]] for c in coords]
        else:
            return [[start_lat, start_lon], [end_lat, end_lon]]
    except:
        return [[start_lat, start_lon], [end_lat, end_lon]]

# MAIN UI
st.markdown('<h1 class="main-header">🚛 VRP Banjir Live Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">6 Algoritma Optimasi + OSM/ESRI Maps + Real Roads + Flood Risk Alert</p>', unsafe_allow_html=True)

if 'locations_df' not in st.session_state:
    default_data = {
        'ID': list(range(11)),
        'Nama': ['Gudang Sentul', 'Cileungsi', 'Gunung Putri', 'Jonggol', 'Cariu', 'Tanjungsari', 'Sukamakmur', 'Klapanunggal', 'Citeureup', 'Babakan Madang', 'Sukaraja'],
        'Latitude': [-6.5546, -6.4035, -6.4398, -6.4716, -6.5869, -6.6163, -6.6080, -6.4780, -6.4859, -6.5744, -6.5644],
        'Longitude': [106.8624, 106.9634, 106.9157, 107.0601, 107.1328, 107.1950, 107.0199, 106.9530, 106.8833, 106.8920, 106.8188],
        'Demand_kg': [0, 1500, 1200, 1000, 800, 700, 600, 900, 1100, 1000, 1300]
    }
    st.session_state.locations_df = pd.DataFrame(default_data)

if 'map_layer' not in st.session_state:
    st.session_state.map_layer = 'osm'

# SIDEBAR - MODERN INPUT
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    
    # TABS untuk organize settings
    tab1, tab2, tab3 = st.tabs(["📍 Locations", "🚚 Vehicle", "⛽ Fuel"])
    
    with tab1:
        st.markdown("**Manage Delivery Points**")
        edited_df = st.data_editor(
            st.session_state.locations_df,
            num_rows="dynamic",
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "Nama": st.column_config.TextColumn("Location Name", width="medium"),
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "Demand_kg": st.column_config.NumberColumn("Demand (kg)", min_value=0)
            },
            use_container_width=True,
            height=400
        )
        st.session_state.locations_df = edited_df
    
    with tab2:
        st.markdown("**Vehicle Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            kapasitas = st.number_input(
                "💪 Truck Capacity",
                min_value=500,
                max_value=20000,
                value=4500,
                step=100,
                help="Maximum load per truck (kg)"
            )
        with col2:
            kecepatan = st.number_input(
                "🚗 Avg Speed",
                min_value=20,
                max_value=100,
                value=50,
                step=5,
                help="Average speed (km/h)"
            )
        
        st.markdown("---")
        st.markdown("**Distribution Costs**")
        col1, col2 = st.columns(2)
        with col1:
            biaya_km = st.number_input(
                "📏 Cost per KM",
                min_value=1000,
                max_value=100000,
                value=12000,
                step=500,
                help="Distribution cost per km (Rp)"
            )
        with col2:
            biaya_jam = st.number_input(
                "⏰ Cost per Hour",
                min_value=10000,
                max_value=200000,
                value=50000,
                step=5000,
                help="Driver cost per hour (Rp)"
            )
    
    with tab3:
        st.markdown("**Fuel Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            fuel_efficiency = st.number_input(
                "⛽ Fuel Efficiency",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help="km per liter"
            )
        with col2:
            fuel_price = st.number_input(
                "💰 Fuel Price",
                min_value=5000,
                max_value=50000,
                value=12500,
                step=500,
                help="Price per liter (Rp)"
            )
        
        fuel_cost_km = fuel_price / fuel_efficiency
        st.metric("📊 Fuel Cost/KM", f"Rp {int(fuel_cost_km):,}")
    
    st.divider()
    st.markdown("**Map Style**")
    map_choice = st.radio(
        "Select map view:",
        ["🌍 OpenStreetMap", "🛰️ ESRI Satellite", "🗺️ OpenTopoMap"],
        horizontal=False,
        label_visibility="collapsed"
    )
    if map_choice == "🌍 OpenStreetMap":
        st.session_state.map_layer = 'osm'
    elif map_choice == "🛰️ ESRI Satellite":
        st.session_state.map_layer = 'esri'
    else:
        st.session_state.map_layer = 'topo'
    
    st.divider()
    if st.button("🚀 OPTIMIZE 6 ALGORITHMS", type="primary", use_container_width=True):
        st.session_state.run_optimization = True
        st.session_state.kapasitas = kapasitas
        st.session_state.kecepatan_rata = kecepatan
        st.session_state.biaya_km = biaya_km + fuel_cost_km
        st.session_state.biaya_jam = biaya_jam
        st.rerun()

# MAIN CONTENT
if st.session_state.get('run_optimization', False):
    with st.spinner("🔄 Running optimization for 6 algorithms..."):
        locations = st.session_state.locations_df.reset_index(drop=True)
        demands = locations['Demand_kg'].tolist()
        dist_matrix = get_distance_matrix(locations)
        weather_data = get_weather_forecast(locations)
        
        solver = VRP_MasterSolver(dist_matrix, demands, st.session_state.kapasitas)
        all_results = {
            'Nearest Neighbor ⚡': solver.solve_nn(),
            'Clarke-Wright ⭐': solver.solve_cw(),
            'Cheapest Insertion 💎': solver.solve_cheapest_insertion(),
            'Nearest Insertion': solver.solve_nearest_insertion(),
            'Farthest Insertion': solver.solve_farthest_insertion(),
            'Arbitrary Insertion': solver.solve_arbitrary_insertion()
        }
        
        method_analysis = {}
        best_method = None
        best_cost = float('inf')
        
        for method_name, routes in all_results.items():
            total_cost = total_dist = total_time = total_flood = 0
            route_details = []
            for r_idx, route in enumerate(routes):
                nodes = [0] + route + [0]
                dist_m = sum(dist_matrix[nodes[j]][nodes[j+1]] for j in range(len(nodes)-1))
                dist_km = dist_m / 1000
                time_h = dist_km / st.session_state.kecepatan_rata
                cost = (dist_km * st.session_state.biaya_km) + (time_h * st.session_state.biaya_jam)
                load = sum(demands[n] for n in route)
                flood_levels = [weather_data.get(n, {}).get('flood_level', '🟢 AMAN') for n in route]
                flood_max = max(flood_levels)
                flood_sc = sum(1 if '🔴' in f else 0.5 if '🟠' in f else 0.25 if '🟡' in f else 0 for f in flood_levels)
                total_cost += cost
                total_dist += dist_km
                total_time += time_h
                total_flood += flood_sc
                names = [locations.iloc[n]['Nama'] for n in nodes]
                route_details.append({'Truk': r_idx + 1, 'Rute': ' → '.join(names), 'Jarak': f"{dist_km:.1f} km", 'Waktu': f"{time_h:.1f} jam", 'Biaya': f"Rp {int(cost):,}", 'Muatan': f"{load:,} kg", 'Banjir': flood_max})
            
            method_analysis[method_name] = {'cost': total_cost, 'dist': total_dist, 'time': total_time, 'truk': len(routes), 'details': route_details, 'routes': routes, 'flood_risk': total_flood / max(1, len(routes))}
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_method = method_name
                st.session_state.best_result = method_analysis[method_name]

    st.markdown("---")
    st.markdown("## 🏆 Optimization Results")
    
    # MODERN METRICS GRID
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card best-method" style="text-align: center; padding: 2rem;">
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.5rem;">🥇 Best Algorithm</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: white; word-break: break-word;">{best_method.split()[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8;">📏 Total Distance</div>
            <div class="metric-value" style="font-size: 1.8rem;">{st.session_state.best_result['dist']:.1f}</div>
            <div style="font-size: 0.8rem; color: #64748b;">km</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8;">💰 Total Cost</div>
            <div class="metric-value" style="font-size: 1.5rem;">Rp {int(best_cost):,}</div>
            <div style="font-size: 0.8rem; color: #64748b;">distribution</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8;">⏱️ Total Time</div>
            <div class="metric-value" style="font-size: 1.8rem;">{st.session_state.best_result['time']:.1f}</div>
            <div style="font-size: 0.8rem; color: #64748b;">hours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8;">🚛 Trucks Needed</div>
            <div class="metric-value" style="font-size: 2rem;">{st.session_state.best_result['truk']}</div>
            <div style="font-size: 0.8rem; color: #64748b;">vehicles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        risk = st.session_state.best_result['flood_risk']
        risk_label = "🟢 LOW" if risk < 0.3 else "🟡 MED" if risk < 0.7 else "🔴 HIGH"
        risk_color = "#10b981" if risk < 0.3 else "#f59e0b" if risk < 0.7 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-color: {risk_color}80; border-left: 4px solid {risk_color};">
            <div style="font-size: 0.85rem; color: #94a3b8;">🌧️ Flood Risk</div>
            <div class="metric-value" style="font-size: 1.5rem; color: {risk_color};">{risk_label}</div>
            <div style="font-size: 0.8rem; color: #64748b;">{risk:.2f}/1</div>
        </div>
        """, unsafe_allow_html=True)
    
    # COMPARISON TABLE
    st.markdown("## 📊 6 Algorithm Comparison")
    comp_data = []
    for rank, (method, data) in enumerate(sorted(method_analysis.items(), key=lambda x: x[1]['cost']), 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        comp_data.append({
            medal: method,
            "💰 Cost": f"Rp {int(data['cost']):,}",
            "📏 Distance": f"{data['dist']:.1f} km",
            "🚛 Trucks": data['truk'],
            "⏱️ Time": f"{data['time']:.1f}h",
            "🌧️ Flood": f"{data['flood_risk']:.2f}"
        })
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, height=300)
    
    # BEST ROUTE DETAILS
    st.markdown("## 🛣️ Best Route Details")
    st.dataframe(pd.DataFrame(st.session_state.best_result['details']), use_container_width=True)
    
    # WEATHER FORECAST
    st.markdown("## 🌧️ Flood Risk Forecast")
    weather_rows = []
    for i, row in locations.iterrows():
        w = weather_data[i]
        weather_rows.append({
            '📍 Location': row['Nama'],
            '📦 Demand': f"{row['Demand_kg']:,} kg",
            '🌧️ Precipitation': f"{w['avg_precip']} mm/h",
            '📊 Probability': f"{w['max_prob']}%",
            '🚨 Status': w['flood_level']
        })
    st.dataframe(pd.DataFrame(weather_rows), use_container_width=True)
    
    # INTERACTIVE MAP
    st.markdown("## 🗺️ Interactive Map - Real Routes")
    
    center_lat = locations['Latitude'].mean()
    center_lon = locations['Longitude'].mean()
    
    routes_list = []
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']
    
    if st.session_state.best_result.get('routes'):
        for route_idx, route in enumerate(st.session_state.best_result['routes']):
            route_nodes = [0] + route + [0]
            route_coords = []
            
            for node_idx in range(len(route_nodes) - 1):
                start_node = route_nodes[node_idx]
                end_node = route_nodes[node_idx + 1]
                
                start_lat = locations.iloc[start_node]['Latitude']
                start_lon = locations.iloc[start_node]['Longitude']
                end_lat = locations.iloc[end_node]['Latitude']
                end_lon = locations.iloc[end_node]['Longitude']
                
                points = get_real_route_osm(start_lat, start_lon, end_lat, end_lon)
                route_coords.extend(points)
            
            routes_list.append({
                'coords': route_coords,
                'color': colors[route_idx % len(colors)],
                'truk': route_idx + 1,
                'detail': st.session_state.best_result['details'][route_idx] if route_idx < len(st.session_state.best_result['details']) else {}
            })
    
    markers_list = []
    for i, row in locations.iterrows():
        w = weather_data.get(i, {})
        color_map = '#dc2626' if '🔴' in w.get('flood_level', '') else '#f97316' if '🟠' in w.get('flood_level', '') else '#eab308' if '🟡' in w.get('flood_level', '') else '#059669'
        
        markers_list.append({
            'lat': row['Latitude'],
            'lon': row['Longitude'],
            'nama': row['Nama'],
            'demand': row['Demand_kg'],
            'precip': w.get('avg_precip', 0),
            'prob': w.get('max_prob', 0),
            'level': w.get('flood_level', '🟢 AMAN'),
            'color': color_map
        })
    
    routes_js = json.dumps(routes_list)
    markers_js = json.dumps(markers_list)
    
    if st.session_state.map_layer == 'esri':
        basemap_js = """
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19,
            attribution: '© Esri'
        }).addTo(map);
        """
        layer_name = "🛰️ ESRI Satellite"
    elif st.session_state.map_layer == 'topo':
        basemap_js = """
        L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            maxZoom: 17,
            attribution: '© OpenTopoMap'
        }).addTo(map);
        """
        layer_name = "🗺️ OpenTopoMap"
    else:
        basemap_js = """
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(map);
        """
        layer_name = "🌍 OpenStreetMap"
    
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Poppins', sans-serif; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
            .info {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(15, 23, 42, 0.95);
                backdrop-filter: blur(10px);
                padding: 18px;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                z-index: 9999;
                font-family: 'Poppins', sans-serif;
                font-size: 13px;
                max-width: 320px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #e2e8f0;
                font-weight: 500;
            }}
            .legend {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(15, 23, 42, 0.95);
                backdrop-filter: blur(10px);
                padding: 18px;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                z-index: 9999;
                font-family: 'Poppins', sans-serif;
                font-size: 12px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #e2e8f0;
            }}
            .legend b {{
                font-weight: 700;
                display: block;
                margin-bottom: 10px;
                font-size: 13px;
            }}
            .legend span {{
                display: block;
                margin: 6px 0;
                font-weight: 600;
            }}
            .info b {{
                font-weight: 700;
                display: block;
                margin-bottom: 8px;
                font-size: 14px;
                background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .info-row {{
                margin: 6px 0;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div class="info">
            <b>{layer_name}</b>
            <div class="info-row">✅ Real roads OSM</div>
            <div class="info-row">📍 Locations: {len(markers_list)}</div>
            <div class="info-row">🚚 Routes: {len(routes_list)}</div>
            <div class="info-row">🌧️ Flood tracking</div>
        </div>
        <div class="legend">
            <b>🌧️ Flood Risk</b>
            <span style="color: #dc2626;">🔴 CRITICAL</span>
            <span style="color: #f97316;">🟠 WARNING</span>
            <span style="color: #eab308;">🟡 ALERT</span>
            <span style="color: #059669;">🟢 SAFE</span>
        </div>
        <script>
            const map = L.map('map').setView([{center_lat}, {center_lon}], 11);
            {basemap_js}
            
            const markers = {markers_js};
            markers.forEach(m => {{
                L.circleMarker([m.lat, m.lon], {{
                    radius: 12,
                    fillColor: m.color,
                    color: 'white',
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 0.85
                }}).bindPopup(`
                    <div style="font-family: Poppins; font-size: 12px; color: #1e293b;">
                        <b style="font-size: 14px; color: {m.color};">${{m.nama}}</b><br>
                        📦 ${{m.demand.toLocaleString()}} kg<br>
                        🌧️ ${{m.precip}} mm/h | ${{m.prob}}%<br>
                        <b style="color: {m.color};">${{m.level}}</b>
                    </div>
                `, {{maxWidth: 250}}).addTo(map);
            }});
            
            const routes = {routes_js};
            routes.forEach((r, idx) => {{
                L.polyline(r.coords, {{
                    color: r.color,
                    weight: 6,
                    opacity: 0.85,
                    lineCap: 'round',
                    lineJoin: 'round'
                }}).bindPopup(`
                    <div style="font-family: Poppins; font-size: 12px; color: #1e293b;">
                        <b style="font-size: 14px; color: {r.color};">🚛 Truk {r.truk}</b><br>
                            📏 {r.detail.get('Jarak', '?.? km')}<br>
                            ⏱️ {r.detail.get('Waktu', '?.? jam')}<br>
                            💰 {r.detail.get('Biaya', 'Rp ?')}<br>
                            📦 {r.detail.get('Muatan', '? kg')}
                    </div>
                `, {{maxWidth: 250}}).addTo(map);
                
                if (r.coords.length > 0) {{
                    const midPoint = r.coords[Math.floor(r.coords.length / 2)];
                    L.marker(midPoint, {{
                        icon: L.divIcon({{
                            html: `<div style="background: {r.color}; color: white; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family: Poppins;">{r.truk}</div>`,
                            iconSize: [36, 36],
                            className: 'custom-marker'
                        }})
                    }}).addTo(map);
                }}
            }});
            
            let group = new L.featureGroup();
            markers.forEach(m => group.addLayer(L.marker([m.lat, m.lon])));
            map.fitBounds(group.getBounds().pad(0.1));
        </script>
    </body>
    </html>
    """
    
    st.components.v1.html(map_html, height=750)
    
    # DOWNLOAD SECTION
    st.markdown("---")
    st.markdown("## 📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = pd.DataFrame(comp_data).to_csv(index=False)
        st.download_button(
            "📊 CSV Comparison",
            csv,
            f"vrp_algorithms_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        def create_excel():
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                pd.DataFrame(comp_data).to_excel(writer, sheet_name='Algorithms', index=False)
                pd.DataFrame(st.session_state.best_result['details']).to_excel(writer, sheet_name='Best Routes', index=False)
                pd.DataFrame(weather_rows).to_excel(writer, sheet_name='Weather', index=False)
                locations.to_excel(writer, sheet_name='Locations', index=False)
            buffer.seek(0)
            return buffer.getvalue()
        
        st.download_button(
            "📈 Excel Report",
            create_excel(),
            f"vrp_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        summary = f"""VRP BANJIR LIVE PRO
═══════════════════════════════════════
Date: {datetime.now().strftime('%d/%m/%Y %H:%M WIB')}
Best Algorithm: {best_method}

METRICS:
• Total Cost: Rp {int(best_cost):,}
• Total Distance: {st.session_state.best_result['dist']:.1f} km
• Total Time: {st.session_state.best_result['time']:.1f} hours
• Trucks: {st.session_state.best_result['truk']}
• Fuel Efficiency: {fuel_efficiency} km/L
• Fuel Price: Rp {fuel_price}/L
• Flood Risk: {st.session_state.best_result['flood_risk']:.2f}/1
"""
        st.download_button(
            "📄 Summary",
            summary,
            f"vrp_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    st.success("✅ Optimization Complete!")
    st.balloons()

else:
    st.info("👈 Configure your settings and click **OPTIMIZE 6 ALGORITHMS** to begin")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; padding: 2rem; font-family: 'Poppins', sans-serif;">
    <p style="margin: 0; font-weight: 600;">🚛 VRP Banjir Live Pro</p>
    <p style="margin: 0; font-size: 0.9rem;">6 Algorithms • OSM/ESRI • Real Roads • Flood Tracking</p>
    <p style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b;">© 2025 - Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
