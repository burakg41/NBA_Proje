import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json
import time

# --- NBA API ---
from nba_api.stats.endpoints import leaguedashplayerstats

# --- AYARLAR ---
SEASON_YEAR = 2025
NBA_SEASON_STRING = '2025-26' 
TARGET_LEAGUE_ID = "61142" 
MY_TEAM_NAME = "Burak's Wizards"
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- YAN PANEL ---
with st.sidebar:
    st.header("Yönetim")
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()
    st.info("Veri Kaynağı: Yahoo + NBA API")
    st.caption("Kapsam: 14 Takım + Top 300 Free Agent")

# --- NBA VERİSİ ---
@st.cache_data(ttl=3600)
def get_nba_real_stats():
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=NBA_SEASON_STRING, per_mode_detailed='PerGame')
        df = stats.get_data_frames()[0]
        nba_data = {}
        for index, row in df.iterrows():
            clean_name = row['PLAYER_NAME'].lower().replace('.', '').strip()
            nba_data[clean_name] = {'GP': row['GP'], 'MPG': row['MIN']}
        return nba_data
    except Exception as e:
        st.warning(f"NBA verisi çekilemedi: {e}")
        return {}

# --- YAHOO VERİSİ ---
@st.cache_data(ttl=3600)
def load_data():
    nba_stats_dict = get_nba_real_stats()
    
    # Secrets Kontrolü
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except:
                st.error("Secrets hatası.")
                return None
        else:
            st.error("Secrets bulunamadı.")
            return None

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
            st.error("Lig bulunamadı.")
            return None

        lg = gm.to_league(target_league_key)
        
        all_data = []
        teams = lg.teams()
        
        # İlerleme Çubuğu Ayarı
        progress_bar = st.progress(0, text="Lig taranıyor...")
        
        # 1. TAKIMLARI TARA
        total_teams = len(teams)
        step = 0
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    for pm, ps in zip(roster, stats):
                        process_player_final(pm, ps, t_name, "Sahipli", all_data, nba_stats_dict)
            except: pass
            
            step += 1
            progress_bar.progress(step / (total_teams + 1), text=f"{t_name} analiz edildi...")

        # 2. FREE AGENT TARA (TOP 300 - Chunk Yöntemi)
        try:
            progress_bar.progress(0.90, text="🆓 300 Free Agent taranıyor (Bu işlem sürebilir)...")
            
            # 300 Oyuncuyu Çek
            fa_players = lg.free_agents(None)[:300]
            fa_ids = [p['player_id'] for p in fa_players]
            
            # Yahoo API çökmemesi için 25'erli paketler halinde soruyoruz
            chunk_size = 25
            for i in range(0, len(fa_ids), chunk_size):
                chunk_ids = fa_ids[i:i + chunk_size]
                chunk_players = fa_players[i:i + chunk_size]
                
                try:
                    chunk_stats = lg.player_stats(chunk_ids, ANALYSIS_TYPE)
                    for pm, ps in zip(chunk_players, chunk_stats):
                        process_player_final(pm, ps, "🆓 FA", "Free Agent", all_data, nba_stats_dict)
                except:
                    pass
                
                # Ufak bir bekleme (API boğulmasın)
                time.sleep(0.1)

        except Exception as e:
            print(e)
            pass

        progress_bar.empty()
        
        if not all_data:
            st.error("Veri yok.")
            return None
            
        return pd.DataFrame(all_data)
        
    except Exception as e:
        st.error(f"Hata: {e}")
        return None

def process_player_final(meta, stat, team_name, ownership, data_list, nba_dict):
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        p_name = meta['name']
        
        # Pozisyon
        raw_pos = meta.get('display_position', '')
        simple_pos = raw_pos.replace('PG', 'G').replace('SG', 'G').replace('SF', 'F').replace('PF', 'F')
        u_pos = list(set(simple_pos.split(',')))
        u_pos.sort(key=lambda x: 1 if x=='G' else (2 if x=='F' else 3))
        final_pos = ",".join(u_pos)

        # NBA Verisi
        c_name = p_name.lower().replace('.', '').strip()
        real_gp = nba_dict.get(c_name, {}).get('GP', 0)
        real_mpg = nba_dict.get(c_name, {}).get('MPG', 0.0)

        # Sakatlık
        st_code = meta.get('status', '')
        if st_code in ['INJ', 'O']: inj = f"🟥 {st_code}"
        elif st_code in ['GTD', 'DTD']: inj = f"Rx {st_code}"
        else: inj = "✅"

        data_list.append({
            'Player': p_name,
            'Team': team_name,
            'Owner_Status': ownership,
            'Pos': final_pos,
            'Health': inj,
            'GP': int(real_gp),
            'MPG': float(real_mpg),
            'FG%': get_val(stat.get('FG%')),
            'FT%': get_val(stat.get('FT%')),
            '3PTM': get_val(stat.get('3PTM')),
            'PTS': get_val(stat.get('PTS')),
            'REB': get_val(stat.get('REB')),
            'AST': get_val(stat.get('AST')),
            'ST': get_val(stat.get('ST')),
            'BLK': get_val(stat.get('BLK')),
            'TO': get_val(stat.get('TO'))
        })
    except: pass

# --- ANALİZ ---
def calculate_z_scores(df):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    if df.empty: return df
    for c in cats:
        if c not in df.columns: df[c] = 0.0
        mean, std = df[c].mean(), df[c].std()
        if std == 0: std = 1
        df[f'z_{c}'] = (mean - df[c]) / std if c == 'TO' else (df[c] - mean) / std
    return df

def analyze_needs(df, my_team):
    z_cols = [f'z_{c}' for c in ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']]
    m_df = df[df['Team'] == my_team]
    if m_df.empty: return [], []
    prof = m_df[z_cols].sum().sort_values()
    return [x.replace('z_', '') for x in prof.head(4).index], [x.replace('z_', '') for x in prof.tail(3).index]

def score_players(df, targets):
    df['Skor'] = 0
    for c in ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']:
        if f'z_{c}' in df.columns:
            w = 3.0 if c in targets else 1.0
            df['Skor'] += df[f'z_{c}'] * w
    return df

# --- ARAYÜZ ---
st.title("🏀 Burak's GM Dashboard")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        c1, c2 = st.columns(2)
        c1.error(f"📉 İhtiyaç: {', '.join(targets)}")
        c2.success(f"📈 Güçlü: {', '.join(strengths)}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        f_stat = col1.multiselect("Filtre:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        h_inj = col2.checkbox("Sakatları Gizle (✅)", value=False)
        
        v_df = df.copy()
        if f_stat: v_df = v_df[v_df['Owner_Status'].isin(f_stat)]
        if h_inj: v_df = v_df[v_df['Health'].str.contains("✅")]

        # GÖSTERİLECEK SÜTUNLAR (HEPSİ)
        all_cols = ['Player', 'Team', 'Pos', 'Health', 'GP', 'MPG', 'Skor', 'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']

        tab1, tab2, tab3 = st.tabs(["🔥 Hedefler", "📋 Kadrom", "🌍 Tüm Liste"])
        
        with tab1:
            trade_df = v_df[v_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                trade_df[all_cols].head(50),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Puan", format="%.1f", max_value=trade_df['Skor'].max()),
                    "GP": st.column_config.NumberColumn("GP", width="small"),
                    "MPG": st.column_config.NumberColumn("MPG", format="%.1f", width="small"),
                    "FG%": st.column_config.NumberColumn("FG%", format="%.1%"),
                    "FT%": st.column_config.NumberColumn("FT%", format="%.1%"),
                    "3PTM": st.column_config.NumberColumn("3PT", format="%.1f"),
                    "PTS": st.column_config.NumberColumn("PTS", format="%.1f"),
                    "REB": st.column_config.NumberColumn("REB", format="%.1f"),
                    "AST": st.column_config.NumberColumn("AST", format="%.1f"),
                    "ST": st.column_config.NumberColumn("ST", format="%.1f"),
                    "BLK": st.column_config.NumberColumn("BLK", format="%.1f"),
                    "TO": st.column_config.NumberColumn("TO", format="%.1f"),
                },
                use_container_width=True
            )
        with tab2:
            st.dataframe(
                df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)[all_cols],
                use_container_width=True
            )
        with tab3:
            st.dataframe(v_df[all_cols], use_container_width=True)
else:
    st.info("Veriler yükleniyor (Top 300 FA biraz sürebilir)...")
