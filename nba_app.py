import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json
import time
import plotly.express as px # Grafik için yeni kütüphane

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
    st.info("Modül: Takas Sihirbazı & Radar")

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
    
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except: return None
        else: return None

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
        
        if not target_league_key: return None

        lg = gm.to_league(target_league_key)
        
        all_data = []
        teams = lg.teams()
        
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
            progress_bar.progress(step / (total_teams + 1))

        # 2. FREE AGENT TARA (TOP 300)
        try:
            progress_bar.progress(0.90, text="🆓 300 Free Agent taranıyor...")
            fa_players = lg.free_agents(None)[:300]
            fa_ids = [p['player_id'] for p in fa_players]
            
            chunk_size = 25
            for i in range(0, len(fa_ids), chunk_size):
                chunk_ids = fa_ids[i:i + chunk_size]
                chunk_players = fa_players[i:i + chunk_size]
                try:
                    chunk_stats = lg.player_stats(chunk_ids, ANALYSIS_TYPE)
                    for pm, ps in zip(chunk_players, chunk_stats):
                        process_player_final(pm, ps, "🆓 FA", "Free Agent", all_data, nba_stats_dict)
                except: pass
                time.sleep(0.1)
        except: pass

        progress_bar.empty()
        return pd.DataFrame(all_data) if all_data else None
        
    except Exception: return None

def process_player_final(meta, stat, team_name, ownership, data_list, nba_dict):
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        p_name = meta['name']
        raw_pos = meta.get('display_position', '')
        simple_pos = raw_pos.replace('PG', 'G').replace('SG', 'G').replace('SF', 'F').replace('PF', 'F')
        u_pos = list(set(simple_pos.split(',')))
        u_pos.sort(key=lambda x: 1 if x=='G' else (2 if x=='F' else 3))
        final_pos = ",".join(u_pos)

        c_name = p_name.lower().replace('.', '').strip()
        real_gp = nba_dict.get(c_name, {}).get('GP', 0)
        real_mpg = nba_dict.get(c_name, {}).get('MPG', 0.0)

        st_code = meta.get('status', '')
        if st_code in ['INJ', 'O']: inj = f"🟥 {st_code}"
        elif st_code in ['GTD', 'DTD']: inj = f"Rx {st_code}"
        else: inj = "✅"

        fg_val = get_val(stat.get('FG%')) * 100
        ft_val = get_val(stat.get('FT%')) * 100

        data_list.append({
            'Player': p_name,
            'Team': team_name,
            'Owner_Status': ownership,
            'Pos': final_pos,
            'Health': inj,
            'GP': int(real_gp),
            'MPG': float(real_mpg),
            'FG%': fg_val, 
            'FT%': ft_val,
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
    
    # Genel Kalite Puanı (Takım yapısından bağımsız, oyuncu ne kadar iyi?)
    df['Genel_Kalite'] = df[[f'z_{c}' for c in cats]].sum(axis=1)
    return df

def analyze_needs(df, my_team):
    z_cols = [f'z_{c}' for c in ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']]
    m_df = df[df['Team'] == my_team]
    if m_df.empty: return [], []
    prof = m_df[z_cols].sum().sort_values()
    return [x.replace('z_', '') for x in prof.head(4).index], [x.replace('z_', '') for x in prof.tail(3).index]

def score_players(df, targets):
    df['Skor'] = 0
    # Dinamik Puan (Senin takımına ne kadar uyuyor?)
    for c in ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']:
        if f'z_{c}' in df.columns:
            w = 3.0 if c in targets else 1.0
            df['Skor'] += df[f'z_{c}'] * w
    return df

# --- TAKAS SİHİRBAZI MANTIĞI ---
def generate_trade_proposals(df, my_team, targets):
    # 1. Benim takımımdan satılacaklar (Skor'u en düşük olanlar = Takıma en az uyanlar)
    my_players = df[df['Team'] == my_team].sort_values(by='Skor', ascending=True) # En kötü uyum en üstte
    
    # 2. Hedefler (Free Agent olmayan, başka takımlardaki oyuncular)
    avail_targets = df[(df['Team'] != my_team) & (df['Owner_Status'] == 'Sahipli')].sort_values(by='Skor', ascending=False)
    
    proposals = []
    
    # Benim en "Gözden Çıkarılabilir" 6 oyuncum ile
    # Ligin bana "En Faydalı" 25 oyuncusunu kıyasla
    for _, my_p in my_players.head(6).iterrows():
        for _, target in avail_targets.head(25).iterrows():
            
            # ADALET KONTROLÜ (FAIRNESS)
            # Kimse LeBron verip vasat oyuncu almaz.
            # Genel_Kalite (Z-Score toplamı) birbirine yakın olmalı.
            quality_diff = target['Genel_Kalite'] - my_p['Genel_Kalite']
            
            # Eğer karşı tarafın oyuncusu çok daha kaliteliyse (Örn: +3 Z-Score farkı) takas reddedilir.
            # Biraz esnek olalım: Kalite farkı +2.0'a kadar olan teklifleri gösterelim (Belki kandırabilirsin).
            if quality_diff < 2.5: 
                gain = target['Skor'] - my_p['Skor']
                if gain > 2.0: # Sadece belirgin bir kazanç varsa öner
                    proposals.append({
                        'Ver': my_p['Player'],
                        'Al': target['Player'],
                        'Hedef Takım': target['Team'],
                        'Takıma Uyum Kazancı': round(gain, 1),
                        'Takas Zorluğu': "Kolay" if quality_diff < 0 else ("Orta" if quality_diff < 1.5 else "Zor (İkna Etmelisin)")
                    })
    
    return pd.DataFrame(proposals)

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

        all_cols = ['Player', 'Team', 'Pos', 'Health', 'GP', 'MPG', 'Skor', 'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']

        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Hedefler", "📋 Kadrom", "🌍 Tüm Liste", "🔄 Takas Sihirbazı"])
        
        with tab1:
            trade_df = v_df[v_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(trade_df[all_cols].head(50), use_container_width=True)
            
            # --- RADAR GRAFİK (SEÇİLEN OYUNCU) ---
            st.subheader("📊 Oyuncu Analizi (Radar)")
            selected_player = st.selectbox("Oyuncu Seç:", trade_df['Player'].head(15))
            
            if selected_player:
                p_data = df[df['Player'] == selected_player].iloc[0]
                categories = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK'] # TO hariç (negatif olduğu için grafiği bozar)
                # Z-Score'ları normalize edip (0-100 arası) grafiğe dökme mantığı (Basit gösterim için Z-Score kullanacağız)
                values = [p_data[f'z_{c}'] for c in categories]
                
                fig = px.line_polar(r=values, theta=categories, line_close=True, title=f"{selected_player} Yetenek Dağılımı")
                fig.update_traces(fill='toself')
                st.plotly_chart(fig)

        with tab2:
            st.dataframe(df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)[all_cols], use_container_width=True)
        with tab3:
            st.dataframe(v_df[all_cols], use_container_width=True)
            
        with tab4:
            st.header("🤖 Akıllı Takas Önerileri")
            st.info("Bu modül, senin takımına en az uyan oyuncuları bulur ve onları, senin eksiklerini kapatan rakip oyuncularla eşleştirir.")
            
            proposals_df = generate_trade_proposals(df, MY_TEAM_NAME, targets)
            
            if not proposals_df.empty:
                st.dataframe(
                    proposals_df,
                    column_config={
                        "Takıma Uyum Kazancı": st.column_config.ProgressColumn("Kazanç", min_value=0, max_value=10, format="+%.1f"),
                    },
                    use_container_width=True
                )
            else:
                st.warning("Şu an mantıklı bir takas önerisi bulunamadı. Free Agent havuzuna bakmanı öneririm.")

else:
    st.info("Sistem başlatılıyor... (İlk açılışta kütüphaneler yüklenirken 1-2 dk sürebilir)")
