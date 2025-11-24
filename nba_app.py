import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json

# --- AYARLAR ---
SEASON_YEAR = 2025
TARGET_LEAGUE_ID = "61142" 
MY_TEAM_NAME = "Burak's Wizards"
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- YAN PANEL ---
with st.sidebar:
    st.header("Yönetim Paneli")
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.info(f"📅 Mod: Sezon Ortalamaları")
    st.caption("Veriler Free Agent havuzunu da içerir.")

# --- VERİ YÜKLEME ---
@st.cache_data(ttl=3600)
def load_data():
    status_container = st.empty()
    debug_container = st.expander("🛠️ Debug Paneli", expanded=False)
    
    # --- BULUT İÇİN GİZLİ DOSYA OLUŞTURMA ---
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except Exception as e:
                st.error(f"Secrets hatası: {e}")
                return None
        else:
            st.error("❌ oauth2.json yok ve Secrets ayarlanmamış!")
            return None
    # ----------------------------------------

    status_container.info("🚀 Yahoo sunucularına bağlanılıyor...")
    
    try:
        sc = OAuth2(None, None, from_file='oauth2.json')
        if not sc.token_is_valid():
            sc.refresh_access_token()
        gm = yfa.Game(sc, 'nba')
        
        league_ids = gm.league_ids(year=SEASON_YEAR)
        target_league_key = None
        for lid in league_ids:
            if TARGET_LEAGUE_ID in lid:
                target_league_key = lid
                break
        
        if not target_league_key:
            st.error(f"❌ Lig ID ({TARGET_LEAGUE_ID}) bulunamadı!")
            return None

        lg = gm.to_league(target_league_key)
        
        # --- VERİ ÇEKME AŞAMASI ---
        all_data = []
        teams = lg.teams()
        
        # İlerleme Çubuğu Ayarı
        # 14 Takım + 1 Free Agent Taraması = Toplam Adım Sayısı
        total_steps = len(teams) + 1 
        progress_bar = st.progress(0, text="Analiz başlıyor...")
        step_count = 0

        # 1. ADIM: TAKIMLARI TARA (SAHİPLİ OYUNCULAR)
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                
                if p_ids:
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    for p_stat in stats:
                        process_player_stats(p_stat, t_name, "Sahipli", all_data)
            except:
                pass
            
            step_count += 1
            progress_bar.progress(step_count / total_steps, text=f"{t_name} tarandı...")

        # 2. ADIM: FREE AGENTLARI TARA (BOŞTAKİLER)
        try:
            progress_bar.progress(0.95, text="🆓 Free Agent havuzu taranıyor (Top 60)...")
            # En iyi 60 boşta oyuncuyu çek
            fa_players = lg.free_agents(None)[:60] 
            fa_ids = [p['player_id'] for p in fa_players]
            
            if fa_ids:
                fa_stats = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                for p_stat in fa_stats:
                    process_player_stats(p_stat, "🆓 FREE AGENT", "Free Agent", all_data)
        except Exception as e:
            debug_container.warning(f"FA Taramasında Hata: {e}")

        progress_bar.empty()
        status_container.empty()
        
        if not all_data:
            st.error("❌ Veri listesi boş!")
            return None
            
        return pd.DataFrame(all_data)
        
    except Exception as e:
        st.error(f"❌ GENEL HATA: {e}")
        return None

def process_player_stats(p_stat, team_name, status_label, data_list):
    """Yardımcı fonksiyon: Veriyi temizleyip listeye ekler"""
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        gp = get_val(p_stat.get('GP'))
        pts = get_val(p_stat.get('PTS'))
        
        # Hiç oynamamış oyuncuyu atla
        if gp == 0 and pts == 0:
            return

        # MPG (Minutes Per Game) Çekimi
        mpg_raw = p_stat.get('MPG', '0')
        if mpg_raw == '-': mpg_raw = '0'
        # Bazen "34:20" gibi string gelir, bazen float. Basite indirgeyeliö.
        try:
            if ":" in str(mpg_raw):
                parts = str(mpg_raw).split(":")
                mpg = float(parts[0]) + (float(parts[1])/60)
            else:
                mpg = float(mpg_raw)
        except:
            mpg = 0.0

        row = {
            'Player': p_stat['name'],
            'Team': team_name,
            'Status': status_label, # Filtreleme için
            'GP': int(gp),          # Maç Sayısı (Tamsayı)
            'MPG': round(mpg, 1),   # Dakika (Virgüllü)
            'FG%': get_val(p_stat.get('FG%')),
            'FT%': get_val(p_stat.get('FT%')),
            '3PTM': get_val(p_stat.get('3PTM')),
            'PTS': pts,
            'REB': get_val(p_stat.get('REB')),
            'AST': get_val(p_stat.get('AST')),
            'ST': get_val(p_stat.get('ST')),
            'BLK': get_val(p_stat.get('BLK')),
            'TO': get_val(p_stat.get('TO'))
        }
        data_list.append(row)
    except:
        pass

def calculate_z_scores(df):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    if df.empty: return df

    for cat in cats:
        if cat not in df.columns: df[cat] = 0.0
        mean = df[cat].mean()
        std = df[cat].std()
        if std == 0: std = 1
        
        col_name = f'z_{cat}'
        if cat == 'TO':
            df[col_name] = (mean - df[cat]) / std
        else:
            df[col_name] = (df[cat] - mean) / std
    return df

def analyze_team_needs(df, my_team_name):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    z_cols = [f'z_{c}' for c in cats]
    
    my_team_df = df[df['Team'] == my_team_name]
    if my_team_df.empty: 
        return [], []

    team_profile = my_team_df[z_cols].sum().sort_values()
    weaknesses = [w.replace('z_', '') for w in team_profile.head(4).index]
    strengths = [s.replace('z_', '') for s in team_profile.tail(3).index]
    
    return weaknesses, strengths

def score_players(df, targets):
    df['Skor'] = 0
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    
    for cat in cats:
        if f'z_{cat}' in df.columns:
            weight = 3.0 if cat in targets else 1.0
            df['Skor'] += df[f'z_{cat}'] * weight
    return df

# --- ARAYÜZ ---

st.title("🏀 Burak's Wizards - GM Paneli")
st.markdown("**Veri Kaynağı:** 2025-2026 Sezonu (GP: Maç Sayısı | MPG: Ortalama Dakika)")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_team_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"📉 **Eksiklerin:** {', '.join(targets)}")
        with col2:
            st.success(f"📈 **Güçlerin:** {', '.join(strengths)}")

        st.markdown("---")
        
        # FİLTRELEME SEÇENEĞİ
        filter_status = st.multiselect(
            "Oyuncu Havuzunu Filtrele:",
            options=["Sahipli", "Free Agent"],
            default=["Sahipli", "Free Agent"]
        )
        
        # Filtreye göre dataframe'i daralt
        if filter_status:
            filtered_df = df[df['Status'].isin(filter_status)]
        else:
            filtered_df = df

        tab1, tab2, tab3 = st.tabs(["🔥 Hedef Oyuncular", "📋 Benim Kadrom", "🌍 Tüm Liste"])

        with tab1:
            st.subheader("Önerilen Oyuncular (Sahipli + Free Agents)")
            st.caption("Eksiklerini en iyi kapatanlar. 'Takım' sütununda 'FREE AGENT' yazanları bedavaya alabilirsin!")
            
            # Kendi takımını hariç tut
            trade_df = filtered_df[filtered_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            
            st.dataframe(
                trade_df[['Player', 'Team', 'GP', 'MPG', 'Skor'] + targets].head(25),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Uygunluk", format="%.1f", min_value=0, max_value=trade_df['Skor'].max()),
                    "Team": st.column_config.TextColumn("Takım / Durum"),
                    "GP": st.column_config.NumberColumn("Maç Sayısı"),
                    "MPG": st.column_config.NumberColumn("Dakika (Ort)", format="%.1f")
                },
                use_container_width=True
            )
            
        with tab2:
            st.subheader("Takım Analizin")
            my_team_df = df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                my_team_df[['Player', 'GP', 'MPG', 'Skor', 'PTS', 'REB', 'AST', 'ST', 'BLK']], 
                use_container_width=True
            )

        with tab3:
            st.dataframe(filtered_df)
            
else:
    st.info("⚠️ Veri bekleniyor...")
