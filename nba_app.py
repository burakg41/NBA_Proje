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
# Tekrar çalışan moda döndük, ama bu sefer hata vermeyecek şekilde ayarladık
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

with st.sidebar:
    st.header("Yönetim")
    if st.button("🔄 Yenile"):
        st.cache_data.clear()
        st.rerun()
    st.info("Mod: Ortalamalar + Debug")

@st.cache_data(ttl=3600)
def load_data():
    debug_container = st.expander("🛠️ Geliştirici / Debug Paneli (Bana Buranın Fotosunu At)", expanded=True)
    
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except:
                return None
        else:
            st.error("Secrets yok!")
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
        
        # --- DEBUG İÇİN İLK VERİYİ YAKALA ---
        first_data_captured = False
        
        total_steps = len(teams) + 1
        progress_bar = st.progress(0, text="Veriler çekiliyor...")
        step = 0

        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                
                if p_ids:
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    
                    # --- DEBUG: İLK OYUNCUNUN VERİSİNİ EKRANA BAS ---
                    if not first_data_captured and stats:
                        debug_container.write("👇 **Yahoo'dan Gelen Ham Veri Örneği (Bunu İncelememiz Lazım):**")
                        debug_container.write(stats[0]) # İlk oyuncunun tüm verisini yaz
                        first_data_captured = True
                    # ------------------------------------------------
                    
                    for player_meta, player_stat in zip(roster, stats):
                        process_player_safe(player_meta, player_stat, t_name, "Sahipli", all_data)
            except:
                pass
            step += 1
            progress_bar.progress(step/total_steps)

        # FA TARAMASI
        try:
            fa_players = lg.free_agents(None)[:40]
            fa_ids = [p['player_id'] for p in fa_players]
            if fa_ids:
                fa_stats = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                for player_meta, player_stat in zip(fa_players, fa_stats):
                    process_player_safe(player_meta, player_stat, "🆓 FREE AGENT", "Free Agent", all_data)
        except:
            pass

        progress_bar.empty()
        
        if not all_data:
            st.error("Veri listesi boş!")
            return None
            
        return pd.DataFrame(all_data)
        
    except Exception as e:
        st.error(f"Hata: {e}")
        return None

def process_player_safe(meta, stat, team_name, ownership, data_list):
    """Hata vermeden ne varsa onu çeken fonksiyon"""
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        # GP ve MPG'yi güvenli çekmeye çalışalım
        gp = 0
        if 'GP' in stat and stat['GP'] != '-': gp = int(stat['GP'])
        
        # MPG / MIN farklı isimlerle gelebilir
        raw_mpg = stat.get('MPG', stat.get('MIN', '0'))
        mpg = 0.0
        try:
            if raw_mpg and raw_mpg != '-':
                s = str(raw_mpg)
                if ":" in s:
                    p = s.split(":")
                    mpg = float(p[0]) + float(p[1])/60
                else:
                    mpg = float(s)
        except:
            mpg = 0.0

        # SAKATLIK
        status = meta.get('status', '')
        inj_display = f"⚠️ {status}" if status else "✅ Sağlam"

        row = {
            'Player': meta['name'],
            'Team': team_name,
            'Owner_Status': ownership,
            'Injury': inj_display,
            'GP': gp,
            'MPG': round(mpg, 1),
            'FG%': get_val(stat.get('FG%')),
            'FT%': get_val(stat.get('FT%')),
            '3PTM': get_val(stat.get('3PTM')),
            'PTS': get_val(stat.get('PTS')),
            'REB': get_val(stat.get('REB')),
            'AST': get_val(stat.get('AST')),
            'ST': get_val(stat.get('ST')),
            'BLK': get_val(stat.get('BLK')),
            'TO': get_val(stat.get('TO'))
        }
        data_list.append(row)
    except:
        pass

# ... (Buradan aşağısı aynı Z-Score ve Arayüz kodları) ...
def calculate_z_scores(df):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    if df.empty: return df
    for cat in cats:
        if cat not in df.columns: df[cat] = 0.0
        mean = df[cat].mean()
        std = df[cat].std()
        if std == 0: std = 1
        col = f'z_{cat}'
        if cat == 'TO': df[col] = (mean - df[cat]) / std
        else: df[col] = (df[cat] - mean) / std
    return df

def analyze_team_needs(df, my_team_name):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    z_cols = [f'z_{c}' for c in cats]
    my_team_df = df[df['Team'] == my_team_name]
    if my_team_df.empty: return [], []
    profile = my_team_df[z_cols].sum().sort_values()
    weak = [w.replace('z_', '') for w in profile.head(4).index]
    strong = [s.replace('z_', '') for s in profile.tail(3).index]
    return weak, strong

def score_players(df, targets):
    df['Skor'] = 0
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    for cat in cats:
        if f'z_{cat}' in df.columns:
            w = 3.0 if cat in targets else 1.0
            df['Skor'] += df[f'z_{cat}'] * w
    return df

st.title("🏀 Burak's GM Dashboard")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_team_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        col1, col2 = st.columns(2)
        col1.error(f"Eksikler: {', '.join(targets)}")
        col2.success(f"Güçler: {', '.join(strengths)}")
        
        st.markdown("---")
        
        col_f1, col_f2 = st.columns(2)
        status_filter = col_f1.multiselect("Tip", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        hide_inj = col_f2.checkbox("Sakatları Gizle", value=False)
        
        filt_df = df[df['Owner_Status'].isin(status_filter)]
        if hide_inj:
            filt_df = filt_df[filt_df['Injury'].str.contains("Sağlam")]

        tab1, tab2, tab3 = st.tabs(["🔥 Hedefler", "📋 Kadrom", "🌍 Tüm Liste"])
        
        with tab1:
            trade_df = filt_df[filt_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                trade_df[['Player', 'Team', 'Injury', 'GP', 'MPG', 'Skor'] + targets].head(30),
                column_config={"Skor": st.column_config.ProgressColumn("Uygunluk", format="%.1f", max_value=trade_df['Skor'].max())},
                use_container_width=True
            )
        with tab2:
            st.dataframe(df[df['Team']==MY_TEAM_NAME].sort_values(by='Skor', ascending=False), use_container_width=True)
        with tab3:
            st.dataframe(filt_df)
else:
    st.warning("Veri bekleniyor...")
