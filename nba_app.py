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
# 🚨 KRİTİK BÖLGE: TOKEN'I BURAYA YAPIŞTIR
# ==========================================
# oauth2.json dosyanın içindeki süslü parantezli tüm yazıyı
# aşağıdaki boşluğa olduğu gibi yapıştır.
# Tırnak hatalarına dikkat et.


MANUAL_TOKEN_DATA = {
    "access_token": "XIO1Yt6au1D4JjQEytWw6fa.gvPesJiOJp.N.ckuYd5ugIsSa44Xv0PQX50MtEqnoSW2l5_7U4QoBD_N174o5aV5FP0yB53w3i4Op_36Ep..g18BwNcSjGjpjD5yZd7c2ThoFR_0GbS.FfQRB80vtPrrIINSqlGC2M1hP1nm4n8bZ2FIj148N85339BL96nWYD7Wl9cJRQKp59bcfiwzSiR2jM9QLwSyY2BQ4PAsbyPAxLDMY2tNnps_SpZ8q7lKMOcRFhImoz0meHJJpKv0jFKKdEFV2osFqHujXkt_lCdgaKYaVXztRpVcP5NUvMRwMFNQIzYi920wPuM0E3PQVY60J0iSert7JZx5BDeOpMQytJyRn3ifSW6Z8I4Nnw989TSqp7g6RzY3X_2K.RP6f5Ilh6tnQqBVGmFghAH8p2RXEcHTQ0doZdNJx6rgqdUZbYLjOVuaJ3aPxhravng4XCNBHmfIXT8puLiBU7wyf_i1VftO.5Spi8wj7s0gPmQ6THG44INVJVn2t83CfWI.J6XDImBTZXoZGLFb1sbDR_CRwJi_ksAeVKc2Z3OuThFRrrzb04UIafrVGeuXbWSX7FVqbtw295k07FD4gBVxt9m7yjknyCusNgO2Rhlp5zT9SEMGc5KR1W4h5kcIFuR6_irgwm2cOJT.J7CZK1oOuUdVFgSHG3fmGPqVUtiu7YxYZo_z6rspctv78HYJG64Olt0r0XNOX6n2HtTGvvycw5y6BTVwemhXObMhaKMWiy4GTc5e.oRouiotNFIntLYD9JpP8t1MMSE5UYi6ETQU6R8Ne.9KHrR6wLAqfP0MAUL_9bPZsj9uHQpkOtNq_5Y2Ukqb1KmiIb2ncmYTriZ99bULdEfp05..FbZKQE95y0qRSNrXEwZ.ZD7.TvGky0fb9MF7bbijhw5MgrX92HSYqDWpE7.5IvPJCP0uv.zcNZG8nd1xHhEbFL_HYdGyTJGCBxs-",
    "consumer_key": "dj0yJmk9SnRUd2xhMzcwWThNJmQ9WVdrOWRHOXlkR1ZaVjJrbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTY5",
    "consumer_secret": "fed1b9a1af2b7d978917ed0d7401578e61ad29f8",
    "refresh_token": "AJ5rJGl_.0az6KK_IHrVgXaM8K.G~001~gz6NHZIiOKCi8PAws78as.8AqODjiPk-",
    "token_time": 1764052012.0662093,
    "token_type": "bearer"
}

# ==========================================
# AYARLAR
# ==========================================
SEASON_YEAR = 2025  
TARGET_LEAGUE_ID = "61142"  
MY_TEAM_NAME = "Burak's Wizards" 
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM v9.5", layout="wide", page_icon="🏀")

# --- NBA API ---
try:
    from nba_api.stats.endpoints import leaguedashplayerstats
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

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
# AUTH VE VERİ FONKSİYONLARI
# ==========================================

def authenticate_direct():
    """
    Dosya veya Secrets okumaz. 
    Doğrudan kodun içindeki MANUAL_TOKEN_DATA değişkenini kullanır.
    """
    # Eğer kullanıcı veriyi yapıştırmamışsa uyarı ver
    if MANUAL_TOKEN_DATA.get("consumer_key") == "BURAYA_YAPISTIR":
        st.error("🚨 Lütfen kodun en üstündeki MANUAL_TOKEN_DATA kısmına oauth2.json içeriğini yapıştırın!")
        st.stop()

    try:
        # Geçici dosya oluştur (Kütüphane dosya istiyor)
        with open('temp_auth.json', 'w') as f:
            json.dump(MANUAL_TOKEN_DATA, f)
        
        sc = OAuth2(None, None, from_file='temp_auth.json')
        
        # Token yenileme denemesi
        if not sc.token_is_valid():
            # Token yenilemeye çalış, olmazsa sessizce devam et (bazen validasyon yanıltır)
            try: sc.refresh_access_token()
            except: pass
            
        return sc
    except Exception as e:
        st.error(f"Auth Hatası: {e}")
        return None

@st.cache_data(ttl=3600)
def get_nba_real_stats():
    """NBA API - Timeout 5 sn. Asla kilitlemez."""
    if not NBA_API_AVAILABLE: return {}
    try:
        custom_headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nba.com/'}
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26', per_mode_detailed='PerGame', timeout=5, headers=custom_headers
        )
        df = stats.get_data_frames()[0]
        nba_data = {}
        for index, row in df.iterrows():
            clean = row['PLAYER_NAME'].lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
            nba_data[clean] = {'GP': row['GP'], 'MPG': row['MIN'], 'TEAM': row['TEAM_ABBREVIATION']}
        return nba_data
    except: return {}

@st.cache_data(ttl=3600)
def get_schedule_robust():
    """Fikstür - ESPN veya Simülasyon."""
    team_counts = {}
    try:
        for i in range(7):
            date_str = (datetime.now() + timedelta(days=i)).strftime('%Y%m%d')
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                for ev in r.json().get('events', []):
                    for comp in ev.get('competitions', []):
                        for c in comp.get('competitors', []):
                            abbr = c['team']['abbreviation']
                            std = TEAM_MAPPER.get(abbr, abbr)
                            team_counts[std] = team_counts.get(std, 0) + 1
            time.sleep(0.05)
    except: pass

    if len(team_counts) < 5: # Fallback
        for t in TEAM_MAPPER.values(): team_counts[t] = 3
    return team_counts

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    status = st.status("Yükleniyor...", expanded=True)
    
    # 1. Auth (En kritik adım)
    status.write("🔐 Giriş yapılıyor...")
    sc = authenticate_direct()
    if not sc: st.stop()

    # 2. Veriler
    status.write("📅 Fikstür ve İstatistikler...")
    nba_schedule = get_schedule_robust()
    nba_stats = get_nba_real_stats()

    # 3. Yahoo
    status.write("📥 Yahoo Verileri...")
    try:
        gm = yfa.Game(sc, 'nba')
        t_lid = next((l for l in gm.league_ids(year=SEASON_YEAR) if TARGET_LEAGUE_ID in l), None)
        if not t_lid: st.error("Lig Bulunamadı"); st.stop()
        
        lg = gm.to_league(t_lid)
        teams = lg.teams()
        all_data = []
        
        prog = status.progress(0)
        for idx, t_key in enumerate(teams.keys()):
            try:
                roster = lg.to_team(t_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    s_s = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    try: s_m = lg.player_stats(p_ids, 'lastmonth')
                    except: s_m = s_s
                    for i, pm in enumerate(roster):
                        if i < len(s_s):
                            m_stat = s_m[i] if i < len(s_m) else s_s[i]
                            process_player(pm, s_s[i], m_stat, teams[t_key]['name'], "Sahipli", all_data, nba_stats, nba_schedule)
            except: pass
            prog.progress((idx + 1) / (len(teams) + 1))
            
        # Free Agents
        try:
            fa_p = lg.free_agents(None)[:60]
            fa_ids = [p['player_id'] for p in fa_p]
            if fa_ids:
                s_s = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                try: s_m = lg.player_stats(fa_ids, 'lastmonth')
                except: s_m = s_s
                for k, pm in enumerate(fa_p):
                    if k < len(s_s):
                        m_stat = s_m[k] if k < len(s_m) else s_s[k]
                        process_player(pm, s_s[k], m_stat, "🆓 FA", "Free Agent", all_data, nba_stats, nba_schedule)
        except: pass
        
        status.update(label="Bitti!", state="complete", expanded=False)
        return pd.DataFrame(all_data), lg
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, None

def process_player(meta, s_s, s_m, t_name, owner, d_list, n_dict, n_sched):
    try:
        def v(x): return float(x) if x not in ['-', None] else 0.0
        name = meta['name']
        pos = "/".join(sorted(list(set(meta.get('display_position','').replace('PG','G').replace('SG','G').replace('SF','F').replace('PF','F').split(',')))))
        
        # NBA Match
        c_name = name.lower().replace('.','').replace("'",'').replace('-',' ').strip()
        gp, mpg, team = 0, 0.0, "N/A"
        if c_name in n_dict:
            gp, mpg, team = n_dict[c_name]['GP'], n_dict[c_name]['MPG'], n_dict[c_name]['TEAM']
        if team == "N/A": 
            team = TEAM_MAPPER.get(meta.get('editorial_team_abbr','').upper(), 'N/A')
            
        g7 = n_sched.get(team, 3)
        
        # Status
        st_c = meta.get('status','')
        inj = "🟥 "+st_c if st_c in ['INJ','O'] else ("Rx "+st_c if st_c in ['GTD','DTD'] else "✅")
        
        # Trend
        def fs(x): return v(x.get('PTS'))+v(x.get('REB'))*1.2+v(x.get('AST'))*1.5+v(x.get('ST'))*3+v(x.get('BLK'))*3-v(x.get('TO'))
        f1, f2 = fs(s_s), fs(s_m)
        diff = f2 - f1
        trend = "➖"
        if "🟥" in inj: trend = "🏥"
        elif abs(diff) > 5.5: trend = "🔥" if diff > 0 else "🥶"
        elif abs(diff) > 2.5: trend = "↗️" if diff > 0 else "↘️"

        d_list.append({
            'Player': name, 'Team': t_name, 'Real_Team': team, 'Owner_Status': owner,
            'Pos': pos, 'Health': inj, 'Trend': trend, 'Games_Next_7D': int(g7),
            'GP': int(gp), 'MPG': float(mpg), 'FG%': v(s_s.get('FG%'))*100, 
            'FT%': v(s_s.get('FT%'))*100, '3PTM': v(s_s.get('3PTM')), 'PTS': v(s_s.get('PTS')),
            'REB': v(s_s.get('REB')), 'AST': v(s_s.get('AST')), 'ST': v(s_s.get('ST')),
            'BLK': v(s_s.get('BLK')), 'TO': v(s_s.get('TO'))
        })
    except: pass

# ==========================================
# ANALİZ
# ==========================================
def get_z(df, punt):
    cats = ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']
    act = [c for c in cats if c not in punt]
    if df.empty: return df, act
    for c in cats:
        if c in punt: df[f'z_{c}'] = 0.0; continue
        m, s = df[c].mean(), df[c].std()
        z = (df[c]-m)/(s if s!=0 else 1)
        df[f'z_{c}'] = -z if c=='TO' else z
    df['Genel_Kalite'] = df[[f'z_{c}' for c in act]].sum(axis=1)
    return df, act

def score_w(df, targets, act):
    df['Skor'] = 0.0
    for c in act:
        w = 1.5 if c in targets else 1.0
        if f'z_{c}' in df.columns: df['Skor'] += df[f'z_{c}'] * w
    # Sakatlık Cezası
    df['Trade_Value'] = df['Skor']
    mask = df['Health'].str.contains('🟥|Rx')
    df.loc[mask, 'Trade_Value'] *= 0.5
    return df

def analyze_needs(df, my_team, act):
    m_df = df[df['Team'].str.strip() == my_team.strip()]
    if m_df.empty: return [], []
    z_cols = [f'z_{c}' for c in act]
    tot = m_df[z_cols].sum().sort_values()
    return [x.replace('z_','') for x in tot.head(3).index], [x.replace('z_','') for x in tot.tail(3).index]

def trade_engine(df, me, opp, needs):
    me, opp = me.strip(), opp.strip()
    my_roster = df[df['Team'].str.strip()==me].sort_values('Trade_Value').head(10)
    op_roster = df[df['Team'].str.strip()==opp].sort_values('Trade_Value', ascending=False).head(10)
    
    groups = {"Küçük": [], "Orta": [], "Büyük": [], "Devasa": []}
    
    for ng in range(1, 5):
        for nr in range(1, 5):
            if abs(ng-nr) > 1: continue
            gn = "Küçük" if ng+nr<=3 else ("Orta" if ng+nr<=5 else ("Büyük" if ng+nr<=7 else "Devasa"))
            
            m_com = list(itertools.combinations(my_roster.index, ng))
            o_com = list(itertools.combinations(op_roster.index, nr))
            if len(m_com)*len(o_com) > 400: m_com, o_com = m_com[:15], o_com[:15]
            
            for mi in m_com:
                for oi in o_com:
                    gl = [df.loc[i] for i in mi]
                    rl = [df.loc[i] for i in oi]
                    
                    vg, vr = sum([p['Trade_Value'] for p in gl]), sum([p['Trade_Value'] for p in rl])
                    net = vr - vg + (len(gl)-len(rl))*0.8
                    
                    if net > 0.5 and (vg-vr) > -4.0:
                        nm = list(set([c for p in rl for c in needs if p.get(f'z_{c}',0)>0.5]))
                        sc = net + len(nm)*1.5
                        warn = "⚠️ SAKAT" if any(["🟥" in p['Health'] for p in rl]) else "Temiz"
                        acc = "🔥 Yüksek" if (vg/vr if vr else 0) > 0.9 else "✅ Orta"
                        
                        groups[gn].append({
                            'Senaryo': f"{len(gl)}v{len(rl)}",
                            'Ver': ", ".join([p['Player'] for p in gl]),
                            'Al': ", ".join([p['Player'] for p in rl]),
                            'Puan': round(sc, 1), 'Durum': warn, 'Şans': acc
                        })
    
    res = {}
    for k, v in groups.items(): res[k] = pd.DataFrame(v).sort_values('Puan', ascending=False) if v else pd.DataFrame()
    return res

# ==========================================
# APP
# ==========================================
st.title("🏀 Burak's GM Dashboard v9.5 (Direct Injection)")

with st.sidebar:
    if st.button("Yenile"): st.cache_data.clear(); st.rerun()
    hide_inj = st.checkbox("Sakatları Gizle")
    punt = st.multiselect("Punt", ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO'])

df, lg = load_data()

if df is not None and not df.empty:
    df['Team'] = df['Team'].astype(str).str.strip()
    df, act = get_z(df, punt)
    weak, strong = analyze_needs(df, MY_TEAM_NAME, act)
    df = score_w(df, weak, act)
    
    v_df = df[~df['Health'].str.contains("🟥")] if hide_inj else df.copy()
    
    c1, c2 = st.columns(2)
    c1.error(f"Zayıf: {', '.join(weak)}")
    c2.success(f"Güçlü: {', '.join(strong)}")
    
    t1, t2, t3 = st.tabs(["Kadro", "Takas", "Rakip"])
    
    with t1:
        tm = st.selectbox("Takım", [MY_TEAM_NAME]+sorted([t for t in df['Team'].unique() if t!=MY_TEAM_NAME]))
        show = v_df[v_df['Team']==tm].sort_values('Skor', ascending=False)
        st.dataframe(show[['Player','Pos','Games_Next_7D','Trend','Health','Skor','GP','MPG','FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']], use_container_width=True, hide_index=True)
        
    with t2:
        ops = sorted([t for t in df['Team'].unique() if t!=MY_TEAM_NAME and t!="Free Agent"])
        op = st.selectbox("Hedef Takım", ops)
        if st.button("Hesapla"):
            res = trade_engine(df, MY_TEAM_NAME, op, weak)
            ts = st.tabs(list(res.keys()))
            for t, (k, d) in zip(ts, res.items()):
                with t:
                    if not d.empty: st.dataframe(d.head(15), use_container_width=True, hide_index=True)
                    else: st.info("Takas yok.")
                    
    with t3:
        op_a = st.selectbox("Rakip Analiz", ops)
        if op_a:
            cats = ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']
            m, o = df[df['Team']==MY_TEAM_NAME][cats].mean(), df[df['Team']==op_a][cats].mean()
            data = []
            sm, so = 0, 0
            for c in cats:
                w = (m[c]<o[c]) if c=='TO' else (m[c]>o[c])
                if w: sm+=1 
                else: so+=1
                data.append({'Kat':c, 'Ben':f"{m[c]:.1f}", 'Rakip':f"{o[c]:.1f}", 'Durum': "✅" if w else "❌"})
            c1, c2 = st.columns(2)
            c1.metric("Skor", f"{sm} - {so}")
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
