import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json
import time
import requests
import plotly.express as px
import itertools 
from datetime import datetime, timedelta

# ==========================================
# AYARLAR & SABİTLER
# ==========================================
SEASON_YEAR = 2025  
NBA_SEASON_STRING = '2025-26' 
TARGET_LEAGUE_ID = "61142"  
MY_TEAM_NAME = "Burak's Wizards" 
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard v9.2", layout="wide", page_icon="🏀")

# --- NBA API ---
try:
    from nba_api.stats.endpoints import leaguedashplayerstats
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# Takım Eşleştirme (Yahoo -> Standart)
TEAM_MAPPER = {
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
    'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GS': 'GSW', 'GSW': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
    'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NO': 'NOP', 'NOP': 'NOP',
    'NY': 'NYK', 'NYK': 'NYK', 'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 
    'PHO': 'PHX', 'PHX': 'PHX', 'POR': 'POR', 'SA': 'SAS', 'SAS': 'SAS', 
    'SAC': 'SAC', 'TOR': 'TOR', 'UTAH': 'UTA', 'UTA': 'UTA', 'WAS': 'WAS', 'WSH': 'WAS'
}

# ==========================================
# 1. VERİ ÇEKME MOTORU (GÜÇLENDİRİLMİŞ)
# ==========================================

@st.cache_data(ttl=3600)
def get_nba_real_stats():
    """
    NBA.com'dan istatistikleri çeker.
    Timeout ve Retry mekanizması eklenmiştir.
    """
    if not NBA_API_AVAILABLE: return {}
    
    # Güçlü Headerlar (Anti-Block)
    custom_headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Referer': 'https://www.nba.com/',
        'Connection': 'keep-alive'
    }

    # 3 Kere Dene
    for attempt in range(3):
        try:
            time.sleep(1) # Nefes al
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=NBA_SEASON_STRING, 
                per_mode_detailed='PerGame',
                timeout=60, # Süreyi artırdık
                headers=custom_headers
            )
            df = stats.get_data_frames()[0]
            nba_data = {}
            for index, row in df.iterrows():
                clean_name = row['PLAYER_NAME'].lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
                nba_data[clean_name] = {'GP': row['GP'], 'MPG': row['MIN'], 'TEAM': row['TEAM_ABBREVIATION']}
            return nba_data
            
        except Exception as e:
            print(f"Deneme {attempt+1} Başarısız: {e}")
            time.sleep(3) # Bekle ve tekrar dene

    return {} # Başaramazsa boş dön (Program çökmesin)

@st.cache_data(ttl=3600)
def get_schedule_robust():
    """Fikstür çeker. Başarısız olursa 'Simülasyon Modu' devreye girer."""
    team_game_counts = {}
    today = datetime.now()
    success_days = 0

    try:
        for i in range(7):
            date_str = (today + timedelta(days=i)).strftime('%Y%m%d')
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
            r = requests.get(url, timeout=5) # ESPN hızlıdır, 5sn yeter
            if r.status_code == 200:
                data = r.json()
                for event in data.get('events', []):
                    for competition in event.get('competitions', []):
                        for competitor in competition.get('competitors', []):
                            abbr = competitor['team']['abbreviation']
                            std_abbr = TEAM_MAPPER.get(abbr, abbr)
                            team_game_counts[std_abbr] = team_game_counts.get(std_abbr, 0) + 1
                success_days += 1
            time.sleep(0.1)
    except: pass

    if success_days < 3 or len(team_game_counts) < 10:
        return _simulate_schedule()
        
    return team_game_counts

def _simulate_schedule():
    """API çalışmazsa her takıma rastgele 3 veya 4 maç atar."""
    sim_counts = {}
    for team in TEAM_MAPPER.values():
        sim_counts[team] = 3 if np.random.rand() > 0.4 else 4
    return sim_counts

def authenticate_yahoo():
    """Cloud Uyumlu Kimlik Doğrulama"""
    if 'yahoo_auth' in st.secrets:
        try:
            data = dict(st.secrets['yahoo_auth'])
            with open('oauth2.json', 'w') as f:
                json.dump(data, f)
        except Exception: pass

    if os.path.exists('oauth2.json'):
        try:
            sc = OAuth2(None, None, from_file='oauth2.json')
            if not sc.token_is_valid():
                sc.refresh_access_token()
            return sc
        except Exception: return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    status = st.status("Veriler Yükleniyor...", expanded=True)
    
    status.write("📅 Fikstür verisi çekiliyor...")
    nba_schedule = get_schedule_robust()
    is_sim = all(v in [3,4] for v in list(nba_schedule.values())[:5])
    if is_sim: status.warning("⚠️ Canlı fikstür alınamadı. Simülasyon modu aktif.")
    else: status.write("✅ Canlı fikstür alındı.")

    status.write("📊 İstatistikler güncelleniyor (Bu işlem 30-40sn sürebilir)...")
    nba_stats_dict = get_nba_real_stats()
    
    if not nba_stats_dict:
        status.warning("⚠️ NBA İstatistikleri çekilemedi (Timeout). Yahoo verileriyle devam ediliyor.")
    
    status.write("🔐 Yahoo Fantasy'ye bağlanılıyor...")
    sc = authenticate_yahoo()
    if not sc: 
        status.update(label="Giriş Hatası", state="error")
        st.error("Kimlik doğrulama başarısız! (Cloud Secrets veya oauth2.json kontrol edin)")
        return None, None

    try:
        gm = yfa.Game(sc, 'nba')
        league_ids = gm.league_ids(year=SEASON_YEAR)
        target_lid = next((lid for lid in league_ids if TARGET_LEAGUE_ID in lid), None)
        
        if not target_lid:
            status.update(label="Lig Bulunamadı", state="error")
            return None, None

        lg = gm.to_league(target_lid)
        teams = lg.teams()
        all_data = []
        
        status.write(f"📥 {len(teams)} Takımın kadrosu indiriliyor...")
        prog = status.progress(0)
        
        for idx, team_key in enumerate(teams.keys()):
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    stats_s = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    try: stats_m = lg.player_stats(p_ids, 'lastmonth')
                    except: stats_m = stats_s 
                    
                    for i, pm in enumerate(roster):
                        if i < len(stats_s):
                            m_stat = stats_m[i] if i < len(stats_m) else stats_s[i]
                            process_player(pm, stats_s[i], m_stat, t_name, "Sahipli", all_data, nba_stats_dict, nba_schedule)
            except: pass
            prog.progress((idx + 1) / (len(teams) + 1))

        status.write("🆓 Free Agents taranıyor...")
        try:
            fa_players = lg.free_agents(None)[:80]
            fa_ids = [p['player_id'] for p in fa_players]
            if fa_ids:
                stats_s = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                try: stats_m = lg.player_stats(fa_ids, 'lastmonth')
                except: stats_m = stats_s
                for k, pm in enumerate(fa_players):
                    if k < len(stats_s):
                        m_stat = stats_m[k] if k < len(stats_m) else stats_s[k]
                        process_player(pm, stats_s[k], m_stat, "🆓 FA", "Free Agent", all_data, nba_stats_dict, nba_schedule)
        except: pass
        
        prog.progress(1.0)
        status.update(label="Hazır!", state="complete", expanded=False)
        return pd.DataFrame(all_data), lg
        
    except Exception as e:
        status.update(label="Hata", state="error")
        st.error(f"Kritik Hata: {e}")
        return None, None

def process_player(meta, stat_s, stat_m, team_name, ownership, data_list, nba_dict, nba_schedule):
    try:
        def val(v):
            if v == '-' or v is None: return 0.0
            try: return float(v)
            except: return 0.0

        name = meta['name']
        
        raw_pos = meta.get('display_position', '').replace('PG','G').replace('SG','G').replace('SF','F').replace('PF','F')
        pos_set = set(raw_pos.split(','))
        order = {'G':1, 'F':2, 'C':3}
        final_pos = "/".join(sorted(list(pos_set), key=lambda x: order.get(x, 9)))

        c_name = name.lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
        real_gp, real_mpg, nba_team = 0, 0.0, "N/A"
        
        if c_name in nba_dict:
            real_gp = nba_dict[c_name]['GP']
            real_mpg = nba_dict[c_name]['MPG']
            nba_team = nba_dict[c_name]['TEAM']
        else:
            y_abbr = meta.get('editorial_team_abbr', 'N/A').upper()
            nba_team = TEAM_MAPPER.get(y_abbr, y_abbr)

        games_7d = nba_schedule.get(nba_team, 0)
        
        st_code = meta.get('status', '')
        is_injured = st_code in ['INJ', 'O', 'GTD', 'DTD']
        inj_str = f"🟥 {st_code}" if st_code in ['INJ', 'O'] else (f"Rx {st_code}" if is_injured else "✅")

        def f_score(s):
            return (val(s.get('PTS')) + val(s.get('REB'))*1.2 + val(s.get('AST'))*1.5 + 
                    val(s.get('ST'))*3 + val(s.get('BLK'))*3 - val(s.get('TO')))
        
        fs_s = f_score(stat_s)
        fs_m = f_score(stat_m)
        trend = "➖"
        if is_injured: trend = "🏥"
        elif abs(fs_s - fs_m) < 0.1: trend = "➖"
        else:
            diff = fs_m - fs_s
            if diff > 5.0: trend = "🔥"
            elif diff > 2.0: trend = "↗️"
            elif diff < -5.0: trend = "🥶"
            elif diff < -2.0: trend = "↘️"

        data_list.append({
            'Player': name, 'Team': team_name, 'Real_Team': nba_team,
            'Owner_Status': ownership, 'Pos': final_pos, 'Health': inj_str, 'Trend': trend,
            'Games_Next_7D': int(games_7d), 'GP': int(real_gp), 'MPG': float(real_mpg),
            'FG%': val(stat_s.get('FG%'))*100, 'FT%': val(stat_s.get('FT%'))*100, 
            '3PTM': val(stat_s.get('3PTM')), 'PTS': val(stat_s.get('PTS')), 
            'REB': val(stat_s.get('REB')), 'AST': val(stat_s.get('AST')), 
            'ST': val(stat_s.get('ST')), 'BLK': val(stat_s.get('BLK')), 
            'TO': val(stat_s.get('TO'))
        })
    except: pass

def calculate_z_scores(df, punt_list):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    active = [c for c in cats if c not in punt_list]
    if df.empty: return df, active
    
    for c in cats:
        if c in punt_list: 
            df[f'z_{c}'] = 0.0
            continue
        mean = df[c].mean(); std = df[c].std()
        if std == 0: std = 1
        z = (df[c] - mean) / std
        if c == 'TO': z = -z 
        df[f'z_{c}'] = z
        
    df['Genel_Kalite'] = df[[f'z_{c}' for c in active]].sum(axis=1)
    return df, active

def score_players_weighted(df, targets, active_cats):
    df['Skor'] = 0.0
    for c in active_cats:
        col = f'z_{c}'
        weight = 1.0
        if c in targets: weight = 1.5 
        if col in df.columns: df['Skor'] += df[col] * weight
            
    df['Trade_Value'] = df['Skor']
    mask_injured = df['Health'].str.contains('🟥|Rx')
    df.loc[mask_injured, 'Trade_Value'] = df.loc[mask_injured, 'Skor'] * 0.5
    return df

def analyze_needs(df, my_team, active_cats):
    m_df = df[df['Team'].str.strip() == my_team.strip()]
    if m_df.empty: return [], []
    z_cols = [f'z_{c}' for c in active_cats]
    totals = m_df[z_cols].sum().sort_values()
    return [x.replace('z_', '') for x in totals.head(3).index], [x.replace('z_', '') for x in totals.tail(3).index]

def trade_engine_grouped(df, my_team, target_opp, my_needs):
    safe_me = my_team.strip()
    safe_opp = target_opp.strip()
    
    my_roster = df[df['Team'].str.strip() == safe_me].sort_values(by='Trade_Value', ascending=True)
    opp_roster = df[df['Team'].str.strip() == safe_opp].sort_values(by='Trade_Value', ascending=False)
    
    my_assets = my_roster.head(10) 
    opp_assets = opp_roster.head(10)
    
    groups = {
        "Küçük (1-2 Oyuncu)": [], "Orta (2-3 Oyuncu)": [], 
        "Büyük (3-4 Oyuncu)": [], "Devasa (4 Oyuncu)": []
    }
    
    for n_give in range(1, 5):
        for n_recv in range(1, 5):
            if abs(n_give - n_recv) > 1: continue
            
            total_p = n_give + n_recv
            if total_p <= 3: g_name = "Küçük (1-2 Oyuncu)"
            elif total_p <= 5: g_name = "Orta (2-3 Oyuncu)"
            elif total_p <= 7: g_name = "Büyük (3-4 Oyuncu)"
            else: g_name = "Devasa (4 Oyuncu)"
            
            my_combos = list(itertools.combinations(my_assets.index, n_give))
            opp_combos = list(itertools.combinations(opp_assets.index, n_recv))
            
            if len(my_combos) * len(opp_combos) > 600:
                my_combos = my_combos[:20]; opp_combos = opp_combos[:20]
            
            for m_idx in my_combos:
                for o_idx in opp_combos:
                    g_list = [df.loc[i] for i in m_idx]
                    r_list = [df.loc[i] for i in o_idx]
                    res = analyze_trade_scenario(g_list, r_list, my_needs)
                    if res: groups[g_name].append(res)
    
    result_dfs = {}
    for g_name, data in groups.items():
        if data: result_dfs[g_name] = pd.DataFrame(data).sort_values(by='Puan', ascending=False)
        else: result_dfs[g_name] = pd.DataFrame()
    return result_dfs

def analyze_trade_scenario(give, recv, my_needs):
    val_give = sum([p['Trade_Value'] for p in give])
    val_recv = sum([p['Trade_Value'] for p in recv])
    slot_adv = (len(give) - len(recv)) * 0.8
    net_diff = val_recv - val_give + slot_adv
    
    if net_diff > 0.5 and (val_give - val_recv) > -4.0:
        needs_met = []
        for p in recv:
            for cat in my_needs:
                if p.get(f'z_{cat}', 0) > 0.5: needs_met.append(cat)
        needs_met = list(set(needs_met))
        strategic_score = net_diff + (len(needs_met) * 1.5)
        
        has_injured = any(["🟥" in p['Health'] for p in recv])
        warn = "⚠️ RİSKLİ (SAKAT)" if has_injured else "Temiz"
        
        g_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in give])
        r_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in recv])
        
        ratio = val_give / val_recv if val_recv != 0 else 0
        acc = "🔥 Çok Yüksek" if ratio > 0.9 else ("✅ Yüksek" if ratio > 0.75 else "🤔 Orta")
        
        return {'Senaryo': f"{len(give)}v{len(recv)}", 'Verilecekler': g_str, 'Alınacaklar': r_str, 'Puan': round(strategic_score, 1), 'Durum': warn, 'Kabul İhtimali': acc}
    return None

st.title("🏀 Burak's GM Dashboard v9.2")

with st.sidebar:
    st.header("Ayarlar")
    if st.button("🔄 Verileri Yenile", type="primary"):
        st.cache_data.clear()
        st.rerun()
    hide_injured = st.checkbox("Sakatları Listeden Gizle", value=False)
    st.markdown("---")
    punt_cats = st.multiselect("Punt:", ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO'])

df, lg = load_data()

if df is not None and not df.empty:
    df['Team'] = df['Team'].astype(str).str.strip()
    df, active_cats = calculate_z_scores(df, punt_cats)
    weak, strong = analyze_needs(df, MY_TEAM_NAME, active_cats)
    df = score_players_weighted(df, weak, active_cats)
    
    view_df = df.copy()
    if hide_injured: view_df = view_df[~view_df['Health'].str.contains("🟥")]

    c1, c2 = st.columns(2)
    c1.error(f"📉 Hedefler: {', '.join(weak)}")
    c2.success(f"📈 Güçlü Yönler: {', '.join(strong)}")
    
    tab1, tab2, tab3 = st.tabs(["📋 Kadrolar & Fikstür", "🤝 Akıllı Takas", "⚔️ Rakip Analizi"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        team_filter = col_a.selectbox("Takım Seç:", [MY_TEAM_NAME] + sorted([t for t in df['Team'].unique() if t != MY_TEAM_NAME]))
        roster_view = view_df[view_df['Team'] == team_filter].sort_values(by='Skor', ascending=False)
        cols_show = ['Player', 'Pos', 'Games_Next_7D', 'Trend', 'Health', 'Skor', 'GP', 'MPG', 'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
        st.dataframe(roster_view[cols_show], column_config={"Skor": st.column_config.ProgressColumn("Değer", format="%.1f", min_value=-5, max_value=15), "Games_Next_7D": st.column_config.NumberColumn("7 Gün", format="%d 🏀"), "Trend": st.column_config.TextColumn("Form")}, use_container_width=True, hide_index=True)
        
    with tab2:
        opponents = sorted([t for t in df['Team'].unique() if t != MY_TEAM_NAME and t != "Free Agent"])
        target_opp = st.selectbox("Rakip Takım:", opponents)
        if st.button("Senaryoları Hesapla"):
            results_dict = trade_engine_grouped(df, MY_TEAM_NAME, target_opp, weak)
            t_s, t_m, t_l, t_h = st.tabs(results_dict.keys())
            for tab, (name, r_df) in zip([t_s, t_m, t_l, t_h], results_dict.items()):
                with tab:
                    if not r_df.empty: st.dataframe(r_df.head(15), column_config={"Puan": st.column_config.ProgressColumn("Stratejik Puan", min_value=0, max_value=12)}, use_container_width=True, hide_index=True)
                    else: st.info(f"Bu grupta ({name}) uygun takas bulunamadı.")

    with tab3:
        opp_anal = st.selectbox("Rakip:", opponents, key="opp_anal")
        if opp_anal:
            cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
            my_s = df[df['Team'] == MY_TEAM_NAME][cats].mean()
            op_s = df[df['Team'] == opp_anal][cats].mean()
            res = []
            s_me, s_op = 0, 0
            for c in cats:
                v_m = my_s[c]; v_o = op_s[c]
                win = (v_m < v_o) if c == 'TO' else (v_m > v_o)
                if win: s_me +=1; res.append({'Kat': c, 'Ben': f"{v_m:.1f}", 'Rakip': f"{v_o:.1f}", 'Kazanan': "✅ Ben"})
                else: s_op +=1; res.append({'Kat': c, 'Ben': f"{v_m:.1f}", 'Rakip': f"{v_o:.1f}", 'Kazanan': "❌ Rakip"})
            c1, c2 = st.columns(2)
            c1.metric("Tahmini Skor", f"{s_me} - {s_op}")
            st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
else:
    st.info("Sistem verileri bekliyor... (Cloud Secrets veya yerel dosya kontrol edin)")
