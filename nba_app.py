import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json

# --- NBA API EKLEMESİ ---
# NBA'den gerçek maç ve dakika verilerini çekmek için
from nba_api.stats.endpoints import leaguedashplayerstats

# --- AYARLAR ---
SEASON_YEAR = 2025
# NBA API Sezon Formatı (2025-26 Sezonu için)
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
    st.info("Veri Kaynağı: Yahoo (Puanlar) + NBA.com (Dakika/Maç)")

# --- NBA VERİSİNİ ÇEKEN FONKSİYON ---
@st.cache_data(ttl=3600)
def get_nba_real_stats():
    """NBA.com'dan tüm oyuncuların gerçek GP ve MPG verilerini çeker"""
    try:
        # Tüm ligin istatistiklerini tek seferde çekiyoruz (Hızlı olması için)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=NBA_SEASON_STRING, per_mode_detailed='PerGame')
        df = stats.get_data_frames()[0]
        
        # Bize sadece İsim, GP (Maç) ve MIN (Dakika) lazım
        # İsimleri standartlaştıralım (Büyük/küçük harf sorunu olmasın diye)
        nba_data = {}
        for index, row in df.iterrows():
            # Oyuncu ismini temizle
            clean_name = row['PLAYER_NAME'].lower().replace('.', '').strip()
            nba_data[clean_name] = {
                'GP': row['GP'],
                'MPG': row['MIN'] # NBA API dakika ortalamasını verir
            }
        return nba_data
    except Exception as e:
        st.warning(f"NBA verileri çekilemedi: {e}")
        return {}

# --- YAHOO VERİ YÜKLEME ---
@st.cache_data(ttl=3600)
def load_data():
    # Önce NBA Verilerini Hazırla
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
        
        total_steps = len(teams) + 1
        progress_bar = st.progress(0, text="Lig taranıyor...")
        step = 0

        # 1. TAKIMLARI TARA
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                
                if p_ids:
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    for player_meta, player_stat in zip(roster, stats):
                        # NBA Sözlüğünü de fonksiyona gönderiyoruz
                        process_player_final(player_meta, player_stat, t_name, "Sahipli", all_data, nba_stats_dict)
            except:
                pass
            step += 1
            progress_bar.progress(step / total_steps)

        # 2. FREE AGENT TARA
        try:
            progress_bar.progress(0.95, text="🆓 Free Agent havuzu taranıyor...")
            fa_players = lg.free_agents(None)[:50]
            fa_ids = [p['player_id'] for p in fa_players]
            if fa_ids:
                fa_stats = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                for player_meta, player_stat in zip(fa_players, fa_stats):
                    process_player_final(player_meta, player_stat, "🆓 FREE AGENT", "Free Agent", all_data, nba_stats_dict)
        except:
            pass

        progress_bar.empty()
        
        if not all_data:
            st.error("Veri listesi boş.")
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

        player_name = meta['name']
        
        # --- NBA VERİSİ EŞLEŞTİRME ---
        # Yahoo'daki ismi temizle (noktaları sil, küçült)
        clean_name = player_name.lower().replace('.', '').strip()
        
        # NBA sözlüğünden bu ismi bulmaya çalış
        real_gp = 0
        real_mpg = 0.0
        
        if clean_name in nba_dict:
            real_gp = nba_dict[clean_name]['GP']
            real_mpg = nba_dict[clean_name]['MPG']
        else:
            # Tam eşleşme yoksa (İsim farklılığı varsa) basit Yahoo verisine dön
            # Ama senin ligde Yahoo bu veriyi vermiyor, o yüzden 0 kalır.
            pass
        # -----------------------------

        # Sakatlık
        status_code = meta.get('status', '')
        if status_code in ['INJ', 'O']: injury_display = f"🟥 {status_code}"
        elif status_code in ['GTD', 'DTD']: injury_display = f"Rx {status_code}"
        else: injury_display = "✅"

        position = meta.get('display_position', '-')

        row = {
            'Player': player_name,
            'Team': team_name,
            'Owner_Status': ownership,
            'Pos': position,
            'Health': injury_display,
            
            # BURAYA DİKKAT: Artık gerçek NBA verisi kullanıyoruz
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
        }
        data_list.append(row)
    except:
        pass

# --- ANALİZ ---
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

# --- ARAYÜZ ---
st.title("🏀 Burak's GM Dashboard")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_team_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        col1, col2 = st.columns(2)
        col1.error(f"📉 İhtiyaçlar: {', '.join(targets)}")
        col2.success(f"📈 Güçlü Yanlar: {', '.join(strengths)}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        f_status = c1.multiselect("Filtre:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        hide_inj = c2.checkbox("Sadece Sağlamlar (✅)", value=False)
        
        view_df = df.copy()
        if f_status: view_df = view_df[view_df['Owner_Status'].isin(f_status)]
        if hide_inj: view_df = view_df[view_df['Health'].str.contains("✅")]

        tab1, tab2, tab3 = st.tabs(["🔥 Hedefler", "📋 Kadrom", "🌍 Tüm Liste"])
        
        with tab1:
            trade_df = view_df[view_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                trade_df[['Player', 'Team', 'Pos', 'Health', 'GP', 'MPG', 'Skor'] + targets].head(30),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Uygunluk", format="%.1f", max_value=trade_df['Skor'].max()),
                    "GP": st.column_config.NumberColumn("Maç"),
                    "MPG": st.column_config.NumberColumn("Dakika", format="%.1f")
                },
                use_container_width=True
            )
        with tab2:
            my_team = df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                my_team[['Player', 'Pos', 'Health', 'GP', 'MPG', 'Skor', 'PTS', 'REB', 'AST', 'ST', 'BLK', '3PTM']], 
                use_container_width=True
            )
        with tab3:
            st.dataframe(view_df)
else:
    st.info("Veriler yükleniyor...")
