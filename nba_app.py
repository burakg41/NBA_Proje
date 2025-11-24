import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json
import time
import plotly.express as px
import itertools 

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
    st.info("Modül: Ultimate Trade Engine (1v1 -> 3v3)")

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
    except Exception: return {}

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

        # 2. FREE AGENT TARA
        try:
            progress_bar.progress(0.90, text="🆓 Free Agent taranıyor...")
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
    for c in ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']:
        if f'z_{c}' in df.columns:
            w = 3.0 if c in targets else 1.0
            df['Skor'] += df[f'z_{c}'] * w
    return df

# --- ULTIMATE TAKAS MOTORU (1-3 Oyuncu Kombinasyonları) ---
def ultimate_trade_engine(df, my_team, targets):
    # Performans için sınırlama şart:
    # Benim takımımdan satılacak en zayıf 7 oyuncu
    my_assets = df[df['Team'] == my_team].sort_values(by='Skor', ascending=True).head(7)
    
    # Rakiplerin listesini al (Her takım için ayrı hesaplayacağız)
    opponents = df[(df['Team'] != my_team) & (df['Owner_Status'] == 'Sahipli')]['Team'].unique()
    
    proposals = []
    
    # İlerleme animasyonu
    prog_bar = st.progress(0, text="Olası takas senaryoları hesaplanıyor...")
    
    for idx, opp_team in enumerate(opponents):
        # Rakibin en iyi 6 oyuncusunu al (Hedef havuzu)
        opp_assets = df[df['Team'] == opp_team].sort_values(by='Skor', ascending=False).head(6)
        
        # Kombinasyon boyutları (1'den 3'e kadar)
        # Sen 1, 2 veya 3 oyuncu verebilirsin
        for n_give in range(1, 4): 
            # Rakip 1, 2 veya 3 oyuncu verebilir
            for n_recv in range(1, 4):
                
                # Benim kombinasyonlarım
                my_combos = list(itertools.combinations(my_assets.index, n_give))
                # Rakibin kombinasyonları
                opp_combos = list(itertools.combinations(opp_assets.index, n_recv))
                
                for m_idxs in my_combos:
                    give_list = [df.loc[i] for i in m_idxs]
                    
                    for o_idxs in opp_combos:
                        recv_list = [df.loc[i] for i in o_idxs]
                        
                        analyze_generic_trade(give_list, recv_list, proposals)
                        
        prog_bar.progress((idx + 1) / len(opponents))
        
    prog_bar.empty()
    return pd.DataFrame(proposals).sort_values(by='Kazanç', ascending=False)

def analyze_generic_trade(give_list, recv_list, proposal_list):
    # Toplam Değerler (Z-Score)
    total_give_val = sum([p['Genel_Kalite'] for p in give_list])
    total_recv_val = sum([p['Genel_Kalite'] for p in recv_list])
    
    # Adalet (Fairness) Kontrolü
    # Rakibin verdiği paket, benim verdiğimden aşırı düşük olmamalı (Ben enayi değilim)
    # Benim verdiğim paket, rakibinkinden aşırı düşük olmamalı (Rakip reddeder)
    val_diff = total_give_val - total_recv_val
    
    # Kabul Aralığı: 
    # val_diff > -4.0 (Rakip biraz daha değerli verebilir, ikna edilebilir)
    if val_diff > -4.0:
        
        # Uyum Skoru (Takıma Katkı)
        total_give_score = sum([p['Skor'] for p in give_list])
        total_recv_score = sum([p['Skor'] for p in recv_list])
        
        gain = total_recv_score - total_give_score
        
        # Eşik Değer: Takasın büyüklüğüne göre kazanç beklentisi artmalı
        threshold = 2.0 + (len(give_list) + len(recv_list)) * 0.5
        
        if gain > threshold:
            give_names = ", ".join([p['Player'] for p in give_list])
            recv_names = ", ".join([p['Player'] for p in recv_list])
            
            # Etki Metni
            impact_text = get_package_impact_text(give_list, recv_list)
            
            # Takas Türü (1v2, 2v3 vb.)
            trade_type = f"{len(give_list)}v{len(recv_list)}"
            
            proposal_list.append({
                'Tür': trade_type,
                'Verilecekler': give_names,
                'Alınacaklar': recv_names,
                'Hedef Takım': recv_list[0]['Team'],
                'Adalet': val_diff,
                'Kazanç': round(gain, 1),
                'Etki': impact_text
            })

def get_package_impact_text(g_list, r_list):
    cats = ['PTS', 'AST', 'REB', 'BLK', 'ST', '3PTM']
    improvements = []
    
    for c in cats:
        give_tot = sum([p[c] for p in g_list])
        # Alınan oyuncuların toplam istatistiği
        recv_tot = sum([p[c] for p in r_list])
        
        # Eğer oyuncu sayısı eşit değilse ortalamayı dengelemek lazım mı?
        # Fantezide genelde "Toplam" katkıya bakılır (Streaming mantığı).
        # Ancak 1 verip 3 alıyorsam, o 2 boş slot için kimi kestiğim önemli.
        # Basitlik için direkt farka bakıyoruz.
        
        diff = recv_tot - give_tot
        if diff > 1.5: improvements.append(f"{c} (+{diff:.1f})")
    
    return f"🚀 {', '.join(improvements)}" if improvements else "Genel İyileşme"

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
            
            st.subheader("📊 Radar Analizi")
            sel_p = st.selectbox("Oyuncu:", trade_df['Player'].head(15))
            if sel_p:
                p_data = df[df['Player'] == sel_p].iloc[0]
                cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK']
                vals = [p_data[f'z_{c}'] for c in cats]
                norm_vals = [max(0, min(100, (v + 3) * 16.6)) for v in vals]
                
                fig = px.line_polar(r=norm_vals, theta=cats, line_close=True, range_r=[0,100], title=sel_p)
                fig.update_traces(fill='toself')
                st.plotly_chart(fig)

        with tab2:
            st.dataframe(df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)[all_cols], use_container_width=True)
        with tab3:
            st.dataframe(v_df[all_cols], use_container_width=True)
            
        with tab4:
            st.header("🧙‍♂️ Ultimate Takas Motoru")
            st.info("Bu sistem 1v1, 1v2, 2v1, 2v3, 3v3 gibi tüm senaryoları senin için hesaplar.")
            
            if st.button("🚀 Olası Senaryoları Hesapla (Biraz zaman alabilir)"):
                prop_df = ultimate_trade_engine(df, MY_TEAM_NAME, targets)
                
                if not prop_df.empty:
                    # Tabloyu daha okunabilir kılmak için
                    st.dataframe(
                        prop_df.head(50), # En iyi 50 öneri
                        column_config={
                            "Adalet": st.column_config.ProgressColumn("Kabul Şansı", min_value=-5, max_value=5, format="%.1f"),
                            "Kazanç": st.column_config.NumberColumn("Takıma Katkı", format="+%.1f ⭐️"),
                            "Tür": st.column_config.TextColumn("Paket Tipi"),
                        },
                        use_container_width=True
                    )
                else:
                    st.warning("Bu kriterlere uygun takas bulunamadı.")
            else:
                st.write("Hesaplamayı başlatmak için butona basın.")

else:
    st.info("Veriler yükleniyor...")
