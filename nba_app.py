import streamlit as st
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
import pandas as pd
import numpy as np
import os
import json
import time
import requests
import itertools 
from datetime import datetime, timedelta

# ==========================================
# 🚨 TOKEN ALANI (GÖMÜLÜ)
# ==========================================
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

st.set_page_config(page_title="Burak's GM v14.0", layout="wide", page_icon="🏀")

# Takım Eşleştirme (ESPN Standart)
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
# 1. AUTH & VERİ ÇEKME
# ==========================================

def authenticate_direct():
    if MANUAL_TOKEN_DATA.get("consumer_key") == "BURAYA_YAPISTIR":
        st.error("🚨 Token hatası! Kodun üstünü kontrol edin.")
        st.stop()
    try:
        with open('temp_auth.json', 'w') as f: json.dump(MANUAL_TOKEN_DATA, f)
        sc = OAuth2(None, None, from_file='temp_auth.json')
        if not sc.token_is_valid():
            try: sc.refresh_access_token()
            except: pass
        return sc
    except Exception as e: st.error(f"Auth Hatası: {e}"); return None

@st.cache_data(ttl=3600)
def get_bball_ref_stats():
    """
    Basketball-Reference sitesinden tüm ligin GP ve MPG verilerini çeker.
    En güvenilir ve detaylı kaynaktır.
    """
    try:
        # 2025 sezonu için (Ekim 2024 - Haziran 2025 arası '2025' olarak geçer)
        url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
        
        # Pandas ile tabloyu oku
        dfs = pd.read_html(url)
        df = dfs[0]
        
        # Veri Temizliği
        # 'Player' sütununda tekrarlar olabilir (takım değiştirenler), 'Tm'='TOT' olanı veya sonuncuyu alacağız.
        # Basitlik için ilkini alıyoruz (çoğunlukla TOT en üsttedir veya sonuncudur).
        df = df[df['Rk'] != 'Rk'] # Başlık tekrarlarını sil
        df = df.drop_duplicates(subset=['Player'], keep='first')
        
        stats_dict = {}
        for _, row in df.iterrows():
            try:
                # İsim temizliği (Aksanları kaldır vs. basit normalize)
                p_name = row['Player'].split("*")[0].strip() # Yıldız işaretini kaldır
                clean_name = p_name.lower().replace('.', '').replace("'", "").replace('-', ' ')
                
                stats_dict[clean_name] = {
                    'GP': int(row['G']),
                    'MPG': float(row['MP'])
                }
            except: continue
            
        return stats_dict
    except Exception as e:
        print(f"B-Ref Hatası: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_schedule_espn():
    """ESPN API (Fikstür)"""
    counts = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for i in range(7):
            d = (datetime.now() + timedelta(days=i)).strftime('%Y%m%d')
            u = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
            r = requests.get(u, headers=headers, timeout=2)
            if r.status_code == 200:
                for e in r.json().get('events', []):
                    for c in e.get('competitions', []):
                        for comp in c.get('competitors', []):
                            abbr = TEAM_MAPPER.get(comp['team']['abbreviation'], comp['team']['abbreviation'])
                            counts[abbr] = counts.get(abbr, 0) + 1
            time.sleep(0.05)
        return counts
    except: return {k: 3 for k in TEAM_MAPPER.values()}

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    st.caption("Sistem Başlatılıyor...")
    sc = authenticate_direct()
    if not sc: st.stop()

    # Paralel Veri Çekimi
    nba_schedule = get_schedule_espn()
    bref_stats = get_bball_ref_stats() # B-Ref verilerini al

    try:
        gm = yfa.Game(sc, 'nba')
        t_lid = next((l for l in gm.league_ids(year=SEASON_YEAR) if TARGET_LEAGUE_ID in l), None)
        if not t_lid: st.error("Lig Bulunamadı"); st.stop()
        
        lg = gm.to_league(t_lid)
        teams = lg.teams()
        all_data = []
        
        prog = st.progress(0, text="Veriler İşleniyor...")
        
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
                            process_player(pm, s_s[i], m_stat, teams[t_key]['name'], "Sahipli", all_data, nba_schedule, bref_stats)
            except: pass
            prog.progress((idx + 1) / (len(teams) + 2))
            
        # Free Agents (Limitli 80)
        try:
            fa_p = lg.free_agents(None)[:80]
            fa_ids = [p['player_id'] for p in fa_p]
            if fa_ids:
                s_s = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                try: s_m = lg.player_stats(fa_ids, 'lastmonth')
                except: s_m = s_s
                for k, pm in enumerate(fa_p):
                    if k < len(s_s):
                        m_stat = s_m[k] if k < len(s_m) else s_s[k]
                        process_player(pm, s_s[k], m_stat, "🆓 FA", "Free Agent", all_data, nba_schedule, bref_stats)
        except: pass
        
        prog.empty()
        return pd.DataFrame(all_data), lg
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, None

def process_player(meta, s_s, s_m, t_name, owner, d_list, n_sched, bref_stats):
    try:
        def v(x): 
            if x in ['-', None]: return 0.0
            try: return float(x)
            except: return 0.0

        name = meta['name']
        
        # --- POZİSYON FİX ---
        raw_pos = meta.get('display_position', '')
        if not raw_pos: raw_pos = meta.get('position_type', 'F')
        if isinstance(raw_pos, list): raw_pos = ",".join(raw_pos)
        u_pos = set(raw_pos.replace('PG','G').replace('SG','G').replace('SF','F').replace('PF','F').split(','))
        order = {'G':1, 'F':2, 'C':3}
        pos = "/".join(sorted([p for p in list(u_pos) if p], key=lambda x: order.get(x, 9)))

        # Takım & Fikstür
        y_abbr = meta.get('editorial_team_abbr','').upper()
        team = TEAM_MAPPER.get(y_abbr, y_abbr)
        g7 = n_sched.get(team, 3) 
        
        # --- GP ve MPG (Basketball-Reference'dan) ---
        # İsim eşleştirme
        c_name = name.lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
        
        # Varsayılan değerler (Eğer B-Ref'te yoksa)
        gp = v(s_s.get('GP')) # Önce Yahoo'dan al
        mpg = 0.0
        
        if c_name in bref_stats:
            gp = bref_stats[c_name]['GP']
            mpg = bref_stats[c_name]['MPG']
        
        # --- SKOR (Verimlilik Puanı) ---
        def calc_fp(stats):
            return (v(stats.get('PTS')) + 
                    v(stats.get('REB'))*1.2 + 
                    v(stats.get('AST'))*1.5 + 
                    v(stats.get('ST'))*3.0 + 
                    v(stats.get('BLK'))*3.0 - 
                    v(stats.get('TO')))
        
        score_season = calc_fp(s_s)
        score_month = calc_fp(s_m)
        
        # --- AKILLI FORM ANALİZİ (Süre ve Maç Bazlı) ---
        diff = score_month - score_season
        
        st_c = meta.get('status','')
        inj = "🟥 "+st_c if st_c in ['INJ','O'] else ("Rx "+st_c if st_c in ['GTD','DTD'] else "✅")
        
        trend = "➖ Nötr"
        
        if "🟥" in inj:
            trend = "🏥 Sakat"
        elif gp < 5: # 5 maçtan az oynadıysa
            trend = "⚠️ Az Maç"
        elif mpg > 0 and mpg < 22: # 22 Dakika altı rotasyon
            trend = "📉 Rotasyon"
        else:
            # Normal Form Hesabı
            if diff >= 6.5: trend = "🔥 Formda"
            elif diff >= 2.5: trend = "↗️ Yükselişte"
            elif diff <= -3.0: trend = "🥶 Düşüşte"
            else: trend = "➖ Nötr"

        d_list.append({
            'Player': name, 'Team': t_name, 'Real_Team': team, 'Owner_Status': owner,
            'Pos': pos, 'Health': inj, 'Trend': trend, 
            'Games_Next_7D': int(g7), 
            'GP': int(gp), 'MPG': mpg,
            'Skor': score_season, 
            'FG%': v(s_s.get('FG%'))*100, 'FT%': v(s_s.get('FT%'))*100, 
            '3PTM': v(s_s.get('3PTM')), 'PTS': v(s_s.get('PTS')),
            'REB': v(s_s.get('REB')), 'AST': v(s_s.get('AST')), 'ST': v(s_s.get('ST')),
            'BLK': v(s_s.get('BLK')), 'TO': v(s_s.get('TO')),
            'Raw_Stats': s_s 
        })
    except Exception as e: print(e)

# ==========================================
# ANALİZ (TRADE MOTORU İÇİN Z-SCORE)
# ==========================================
def get_z_and_trade_val(df, punt):
    cats = ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']
    act = [c for c in cats if c not in punt]
    if df.empty: return df, act
    
    for c in cats:
        if c in punt: 
            df[f'z_{c}'] = 0.0
            continue
        m, s = df[c].mean(), df[c].std()
        z = (df[c]-m)/(s if s!=0 else 1)
        df[f'z_{c}'] = -z if c=='TO' else z
        
    df['Trade_Value'] = df[[f'z_{c}' for c in act]].sum(axis=1)
    # Sakatlık Cezası (%50)
    mask = df['Health'].str.contains('🟥|Rx')
    df.loc[mask, 'Trade_Value'] *= 0.5
    return df, act

def analyze_needs(df, my_team, act):
    m_df = df[df['Team'].str.strip() == my_team.strip()]
    if m_df.empty: return [], []
    z_cols = [f'z_{c}' for c in act]
    tot = m_df[z_cols].sum().sort_values()
    return [x.replace('z_','') for x in tot.head(3).index], [x.replace('z_','') for x in tot.tail(3).index]

def trade_engine_grouped(df, my_team, target_opp, my_needs):
    safe_me = my_team.strip()
    safe_opp = target_opp.strip()
    
    my_roster = df[df['Team'].str.strip() == safe_me].sort_values(by='Trade_Value', ascending=True)
    opp_roster = df[df['Team'].str.strip() == safe_opp].sort_values(by='Trade_Value', ascending=False)
    
    my_assets = my_roster.head(12) 
    opp_assets = opp_roster.head(12)
    
    groups = {"Küçük (1-2)": [], "Orta (2-3)": [], "Büyük (3-4)": [], "Devasa (4)": []}
    
    for ng in range(1, 5):
        for nr in range(1, 5):
            if abs(ng - nr) > 2: continue
            
            total_p = ng + nr
            if total_p <= 3: g_name = "Küçük (1-2)"
            elif total_p <= 5: g_name = "Orta (2-3)"
            elif total_p <= 7: g_name = "Büyük (3-4)"
            else: g_name = "Devasa (4)"
            
            my_combos = list(itertools.combinations(my_assets.index, ng))
            opp_combos = list(itertools.combinations(opp_assets.index, nr))
            
            if len(my_combos) * len(opp_combos) > 600:
                my_combos, opp_combos = my_combos[:20], opp_combos[:20]
            
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
    slot_adv = (len(give) - len(recv)) * 0.5
    net_diff = val_recv - val_give + slot_adv
    
    if net_diff > 0.5 and (val_give - val_recv) > -4.0:
        needs_met = []
        for p in recv:
            for cat in my_needs:
                if p.get(f'z_{cat}', 0) > 0.5: needs_met.append(cat)
        needs_met = list(set(needs_met))
        strategic_score = net_diff + (len(needs_met) * 1.2)
        
        has_injured = any(["🟥" in p['Health'] for p in recv])
        warn = "⚠️ RİSKLİ" if has_injured else "Temiz"
        
        g_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in give])
        r_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in recv])
        
        ratio = val_give / val_recv if val_recv != 0 else 0
        acc = "🔥 Yüksek" if ratio > 0.9 else ("✅ Orta" if ratio > 0.75 else "🤔 Düşük")
        
        return {'Senaryo': f"{len(give)}v{len(recv)}", 'Verilecekler': g_str, 'Alınacaklar': r_str, 'Puan': round(strategic_score, 1), 'Durum': warn, 'Şans': acc}
    return None

# ==========================================
# APP UI
# ==========================================
st.title("🏀 Burak's GM Dashboard v14.0")

with st.sidebar:
    if st.button("Yenile"): st.cache_data.clear(); st.rerun()
    hide_inj = st.checkbox("Sakatları Gizle")
    punt = st.multiselect("Punt", ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO'])

df, lg = load_data()

if df is not None and not df.empty:
    df['Team'] = df['Team'].astype(str).str.strip()
    
    df, act = get_z_and_trade_val(df, punt)
    weak, strong = analyze_needs(df, MY_TEAM_NAME, act)
    
    v_df = df[~df['Health'].str.contains("🟥")] if hide_inj else df.copy()
    
    c1, c2 = st.columns(2)
    c1.error(f"Hedefler: {', '.join(weak)}")
    c2.success(f"Güçlü: {', '.join(strong)}")
    
    t1, t2, t3 = st.tabs(["Kadro", "Takas", "Rakip"])
    
    with t1:
        tm = st.selectbox("Takım", [MY_TEAM_NAME]+sorted([t for t in df['Team'].unique() if t!=MY_TEAM_NAME]))
        show = v_df[v_df['Team']==tm].sort_values('Skor', ascending=False)
        st.dataframe(
            show[['Player','Pos','Games_Next_7D','Trend','Health','Skor','GP','MPG','FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']], 
            column_config={
                "Skor": st.column_config.ProgressColumn("Verim Puanı", format="%.1f", min_value=0, max_value=60),
                "Trend": st.column_config.TextColumn("Form"),
                "MPG": st.column_config.NumberColumn("Dakika", format="%.1f")
            },
            use_container_width=True, 
            hide_index=True
        )
        
    with t2:
        ops = sorted([t for t in df['Team'].unique() if t!=MY_TEAM_NAME and t!="Free Agent"])
        op = st.selectbox("Hedef Takım", ops)
        if st.button("Hesapla"):
            res = trade_engine_grouped(df, MY_TEAM_NAME, op, weak)
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
else:
    st.info("Veri bekleniyor...")
