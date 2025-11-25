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

# NBA API (GP & MPG için)
try:
    from nba_api.stats.endpoints import leaguedashplayerstats
except ImportError:
    leaguedashplayerstats = None

# ==========================================
# 🚨 TOKEN ALANI (DOLU VE HAZIR)
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
ANALYSIS_TYPE_AVG = 'average_season' 
ANALYSIS_TYPE_TOTAL = 'season'

NBA_SEASON_STRING = "2025-26"

# Cache'i kırmak için versiyon anahtarı
DATA_VERSION = "v19_matchup_fallback"

st.set_page_config(page_title="Burak's GM v15.0", layout="wide", page_icon="🏀")

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
# HELPER: İSİM NORMALİZASYONU
# ==========================================
def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = name.upper()
    for ch in [".", "'", "`"]:
        n = n.replace(ch, "")
    for suf in [" JR", " SR", " III", " II", " IV"]:
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = " ".join(n.split())
    return n

# ==========================================
# AUTH & SCHEDULE
# ==========================================
def authenticate_direct():
    if MANUAL_TOKEN_DATA.get("consumer_key") == "BURAYA_YAPISTIR":
        st.error("🚨 Token hatası!")
        st.stop()
    try:
        with open('temp_auth.json', 'w') as f:
            json.dump(MANUAL_TOKEN_DATA, f)
        sc = OAuth2(None, None, from_file='temp_auth.json')
        if not sc.token_is_valid():
            try:
                sc.refresh_access_token()
            except Exception:
                pass
        return sc
    except Exception as e:
        st.error(f"Auth Hatası: {e}")
        return None

@st.cache_data(ttl=3600)
def get_schedule_espn():
    """Takım bazlı, 7 günlük toplam maç sayısı."""
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
    except Exception:
        return {k: 3 for k in TEAM_MAPPER.values()}

@st.cache_data(ttl=900)
def get_schedule_espn_by_day():
    """
    Önümüzdeki 7 gün için, gün bazlı NBA fikstürü.
    return: { 'YYYY-MM-DD': { 'BOS': 1, 'LAL': 1, ...}, ... }
    """
    result = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for i in range(7):
            date_obj = datetime.now().date() + timedelta(days=i)
            key_day = date_obj.strftime('%Y-%m-%d')
            key_api = date_obj.strftime('%Y%m%d')
            result[key_day] = {}
            u = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={key_api}"
            r = requests.get(u, headers=headers, timeout=2)
            if r.status_code == 200:
                for e in r.json().get('events', []):
                    for c in e.get('competitions', []):
                        for comp in c.get('competitors', []):
                            abbr = TEAM_MAPPER.get(comp['team']['abbreviation'], comp['team']['abbreviation'])
                            result[key_day][abbr] = result[key_day].get(abbr, 0) + 1
            time.sleep(0.05)
        return result
    except Exception:
        # Hata durumunda boş dict döner
        return {}

@st.cache_data(ttl=3600)
def get_nba_base_stats():
    if leaguedashplayerstats is None:
        print("nba_api yüklü değil.")
        return None
    try:
        resp = leaguedashplayerstats.LeagueDashPlayerStats(
            season=NBA_SEASON_STRING,
            per_mode_detailed='PerGame'
        )
        df = resp.get_data_frames()[0]
        df = df[['PLAYER_NAME', 'GP', 'MIN']].copy()
        df['norm_name'] = df['PLAYER_NAME'].apply(normalize_name)
        return df
    except Exception as e:
        print("nba_api error:", e)
        return None

# ==========================================
# PLAYER PROCESS
# ==========================================
def process_player(meta, s_avg, s_total, s_m, t_name, owner, d_list, n_sched, nba_df):
    try:
        def v(x): 
            if x in ['-', None]:
                return 0.0
            try:
                return float(x)
            except Exception:
                return 0.0
        
        def parse_mpg(val):
            if not val or val == '-':
                return 0.0
            try:
                val_str = str(val)
                if ':' in val_str:
                    m, s = map(int, val_str.split(':'))
                    return round(m + s/60, 1)
                return float(val_str)
            except Exception:
                return 0.0

        name = meta['name']
        
        # Pozisyon
        raw_pos = meta.get('display_position') or meta.get('eligible_positions') or ''
        if isinstance(raw_pos, list):
            pos_list = [str(p).strip() for p in raw_pos if p]
        else:
            pos_list = [p.strip() for p in str(raw_pos).replace(' ', '').split(',') if p]

        if len(pos_list) == 0:
            final_pos = 'UTL'
        elif len(pos_list) == 1:
            final_pos = pos_list[0]
        else:
            final_pos = "/".join(pos_list[:2])

        # GP & MPG (Yahoo + NBA override)
        gp = v(
            s_total.get('GP')
            or s_total.get('G')
            or s_avg.get('GP')
            or s_avg.get('G')
        )
        total_min_raw = s_total.get('Min') or s_total.get('MIN')
        total_min = v(total_min_raw)

        if gp > 0 and total_min > 0:
            mpg = round(total_min / gp, 1)
        else:
            avg_min_raw = s_avg.get('Min') or s_avg.get('MIN')
            mpg = parse_mpg(avg_min_raw)

        # NBA override
        if nba_df is not None:
            nname = normalize_name(name)
            row = nba_df[nba_df['norm_name'] == nname]
            if row.empty and len(name.split()) >= 2:
                first, last = name.split()[0], name.split()[-1]
                tmp = nba_df[
                    nba_df['PLAYER_NAME'].str.contains(first, case=False, na=False)
                    & nba_df['PLAYER_NAME'].str.contains(last, case=False, na=False)
                ]
                if not tmp.empty:
                    row = tmp
            if not row.empty:
                r0 = row.iloc[0]
                gp_n = v(r0['GP'])
                mpg_n = v(r0['MIN'])
                if gp_n > 0:
                    gp = int(gp_n)
                    mpg = float(mpg_n)

        # Takım & fikstür
        y_abbr = meta.get('editorial_team_abbr','').upper()
        team = TEAM_MAPPER.get(y_abbr, y_abbr)
        g7 = n_sched.get(team, 3) 
        
        # Skor (fantasy verim puanı)
        def calc_fp(stats):
            return (
                v(stats.get('PTS')) +
                v(stats.get('REB')) * 1.2 +
                v(stats.get('AST')) * 1.5 +
                v(stats.get('ST')) * 3.0 +
                v(stats.get('BLK')) * 3.0 -
                v(stats.get('TO'))
            )
        
        score_season = calc_fp(s_avg)
        score_month = calc_fp(s_m)
        
        # Sağlık
        raw_status = str(meta.get('status', '') or '').upper()
        if raw_status in ['O', 'OUT', 'IR', 'IL', 'IL+', 'INJ']:
            inj = f"🔴 Sakat ({raw_status})"
        elif raw_status in ['GTD', 'DTD', 'NA', 'DAY_TO_DAY']:
            inj = f"🟠 Riskli ({raw_status})"
        else:
            inj = "🟢 Sağlıklı"

        # Form
        if "🔴" in inj:
            trend = "🔴 Sakat"
        else:
            if gp < 5 or score_season < 5:
                trend = "⚪ Verisiz"
            else:
                if mpg >= 32 and score_season >= 32:
                    trend = "🟣 Aşırı Formda"
                elif 28 <= mpg < 32 and 26 <= score_season < 32:
                    trend = "🟢 Formda"
                elif mpg > 18 and score_season > 20:
                    trend = "🟡 Yükselişte"
                else:
                    if 24 >= mpg >= 18:
                        trend = "🔵 Rotasyon Oyuncusu"
                    elif 18 > mpg >= 10:
                        trend = "🟤 Az Oynayan"
                    elif 10 > mpg > 0:
                        trend = "⬛ Nadir Oynayan"
                    else:
                        trend = "⚪ Verisiz"

        d_list.append({
            'Player': name,
            'Team': t_name,
            'Real_Team': team,
            'Owner_Status': owner,
            'Pos': final_pos,
            'Health': inj,
            'Trend': trend, 
            'Games_Next_7D': int(g7), 
            'GP': int(gp),
            'MPG': mpg,
            'Skor': score_season, 
            'FG%': v(s_avg.get('FG%'))*100,
            'FT%': v(s_avg.get('FT%'))*100, 
            '3PTM': v(s_avg.get('3PTM')),
            'PTS': v(s_avg.get('PTS')),
            'REB': v(s_avg.get('REB')),
            'AST': v(s_avg.get('AST')),
            'ST': v(s_avg.get('ST')),
            'BLK': v(s_avg.get('BLK')),
            'TO': v(s_avg.get('TO')),
            'Raw_Stats': s_avg
        })
    except Exception as e:
        print("process_player error:", e)

# DATA_VERSION cache key'e girsin diye parametreli
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(version: str):
    st.caption(f"Sistem Başlatılıyor... (ver: {version})")
    sc = authenticate_direct()
    if not sc:
        st.stop()

    nba_schedule = get_schedule_espn()
    nba_df = get_nba_base_stats()

    try:
        gm = yfa.Game(sc, 'nba')
        t_lid = next((l for l in gm.league_ids(year=SEASON_YEAR) if TARGET_LEAGUE_ID in l), None)
        if not t_lid:
            st.error("Lig Bulunamadı")
            st.stop()
        
        lg = gm.to_league(t_lid)
        teams = lg.teams()
        all_data = []
        
        prog = st.progress(0, text="Veriler İndiriliyor...")
        
        for idx, t_key in enumerate(teams.keys()):
            try:
                roster = lg.to_team(t_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    s_avg = lg.player_stats(p_ids, ANALYSIS_TYPE_AVG)
                    s_total = lg.player_stats(p_ids, ANALYSIS_TYPE_TOTAL)
                    try:
                        s_m = lg.player_stats(p_ids, 'lastmonth')
                    except Exception:
                        s_m = s_avg
                    
                    for i, pm in enumerate(roster):
                        if i < len(s_avg):
                            m_stat = s_m[i] if i < len(s_m) else s_avg[i]
                            t_stat = s_total[i] if i < len(s_total) else s_avg[i]
                            
                            process_player(
                                pm,
                                s_avg[i],
                                t_stat,
                                m_stat,
                                teams[t_key]['name'],
                                "Sahipli",
                                all_data,
                                nba_schedule,
                                nba_df
                            )
            except Exception as e:
                print("team loop error:", e)
            prog.progress((idx + 1) / (len(teams) + 2))
            
        # Free Agents
        try:
            fa_p = lg.free_agents(None)[:80]
            fa_ids = [p['player_id'] for p in fa_p]
            if fa_ids:
                s_avg = lg.player_stats(fa_ids, ANALYSIS_TYPE_AVG)
                s_total = lg.player_stats(fa_ids, ANALYSIS_TYPE_TOTAL)
                try:
                    s_m = lg.player_stats(fa_ids, 'lastmonth')
                except Exception:
                    s_m = s_avg
                
                for k, pm in enumerate(fa_p):
                    if k < len(s_avg):
                        m_stat = s_m[k] if k < len(s_m) else s_avg[k]
                        t_stat = s_total[k] if k < len(s_total) else s_avg[k]
                        process_player(
                            pm,
                            s_avg[k],
                            t_stat,
                            m_stat,
                            "🆓 FA",
                            "Free Agent",
                            all_data,
                            nba_schedule,
                            nba_df
                        )
        except Exception as e:
            print("FA loop error:", e)
        
        prog.empty()
        return pd.DataFrame(all_data), lg
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, None

# ==========================================
# ANALİZ (Z-SCORE, TAKAS)
# ==========================================
def get_z_and_trade_val(df, punt):
    cats = ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']
    act = [c for c in cats if c not in punt]
    if df.empty:
        return df, act
    
    for c in cats:
        if c in punt:
            df[f'z_{c}'] = 0.0
            continue
        m, s = df[c].mean(), df[c].std()
        z = (df[c] - m) / (s if s != 0 else 1)
        df[f'z_{c}'] = -z if c == 'TO' else z
        
    df['Uygunluk_Puanı'] = df[[f'z_{c}' for c in act]].sum(axis=1)
    mask = df['Health'].astype(str).str.contains('Sakat|Riskli', regex=True, na=False)
    df.loc[mask, 'Uygunluk_Puanı'] *= 0.5
    
    return df, act

def analyze_needs(df, my_team, act):
    m_df = df[df['Team'].str.strip() == my_team.strip()]
    if m_df.empty:
        return [], []
    z_cols = [f'z_{c}' for c in act]
    tot = m_df[z_cols].sum().sort_values()
    return [x.replace('z_','') for x in tot.head(3).index], [x.replace('z_','') for x in tot.tail(3).index]

def analyze_trade_scenario(give, recv, my_needs):
    val_give = sum([p['Uygunluk_Puanı'] for p in give])
    val_recv = sum([p['Uygunluk_Puanı'] for p in recv])
    slot_adv = (len(give) - len(recv)) * 0.5
    net_diff = val_recv - val_give + slot_adv  # senin net kazancın

    if net_diff > 0.5 and (val_give - val_recv) > -4.0:
        example = give[0]
        z_cols = [c for c in example.index if c.startswith('z_')]
        cat_impacts = []
        for zc in z_cols:
            cat = zc.replace('z_', '')
            sum_recv = sum([p.get(zc, 0.0) for p in recv])
            sum_give = sum([p.get(zc, 0.0) for p in give])
            delta = sum_recv - sum_give
            cat_impacts.append((cat, delta))

        cat_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_impacts = cat_impacts[:4]
        etkistr = ", ".join([f"{c}:{d:+.2f}" for c, d in top_impacts])

        needs_met = list(set([
            c for p in recv for c in my_needs
            if p.get(f'z_{c}', 0) > 0.5
        ]))
        strategic_score = net_diff + (len(needs_met) * 1.2)
        
        has_injured = any(["Sakat" in p['Health'] or "Riskli" in p['Health'] for p in recv])
        warn = "⚠️ RİSKLİ" if has_injured else "Temiz"
        
        g_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in give])
        r_str = ", ".join([f"{p['Player']} ({p['Pos']})" for p in recv])
        
        ratio = val_give / val_recv if val_recv != 0 else 0
        if ratio < 0.6 or ratio > 1.4:
            acc = "⚪ Düşük"
        else:
            if net_diff >= 2.0:
                acc = "🔥 Çok Yüksek"
            elif net_diff >= 1.0:
                acc = "✅ Yüksek"
            else:
                acc = "🟡 Orta"
        
        return {
            'Senaryo': f"{len(give)}v{len(recv)}",
            'Verilecekler': g_str,
            'Alınacaklar': r_str,
            'Uygunluk_Puanı': round(strategic_score, 1),
            'Kategori_Etkisi': etkistr,
            'Durum': warn,
            'Şans': acc
        }
    return None

def trade_engine_grouped(df, my_team, target_opp, my_needs):
    safe_me = my_team.strip()
    safe_opp = target_opp.strip()
    my_roster = df[df['Team'].str.strip() == safe_me].sort_values(by='Uygunluk_Puanı', ascending=True)
    opp_roster = df[df['Team'].str.strip() == safe_opp].sort_values(by='Uygunluk_Puanı', ascending=False)
    my_assets = my_roster.head(10)
    opp_assets = opp_roster.head(10)
    groups = {
        "Küçük Paket (1-2 Oyuncu)": [],
        "Orta Paket (2-3 Oyuncu)": [],
        "Büyük Paket (3-4 Oyuncu)": [],
        "Devasa Paket (4+ Oyuncu)": []
    }
    
    for ng in range(1, 5):
        for nr in range(1, 5):
            if abs(ng - nr) > 2:
                continue
            total_p = ng + nr
            if total_p <= 3:
                g_name = "Küçük Paket (1-2 Oyuncu)"
            elif total_p <= 5:
                g_name = "Orta Paket (2-3 Oyuncu)"
            elif total_p <= 7:
                g_name = "Büyük Paket (3-4 Oyuncu)"
            else:
                g_name = "Devasa Paket (4+ Oyuncu)"
            
            my_combos = list(itertools.combinations(my_assets.index, ng))
            opp_combos = list(itertools.combinations(opp_assets.index, nr))
            
            if len(my_combos) * len(opp_combos) > 600:
                my_combos, opp_combos = my_combos[:15], opp_combos[:15]
            
            for m_idx in my_combos:
                for o_idx in opp_combos:
                    g_list = [df.loc[i] for i in m_idx]
                    r_list = [df.loc[i] for i in o_idx]
                    res = analyze_trade_scenario(g_list, r_list, my_needs)
                    if res:
                        groups[g_name].append(res)
    
    result_dfs = {}
    for g_name, data in groups.items():
        if data:
            result_dfs[g_name] = pd.DataFrame(data).sort_values(by='Uygunluk_Puanı', ascending=False)
        else:
            result_dfs[g_name] = pd.DataFrame()
    return result_dfs

# ==========================================
# HAFTALIK EŞLEŞME PROJEKSİYONU (ESPN + FALLBACK)
# ==========================================
def project_team_week(team_df: pd.DataFrame, sched_by_day: dict):
    """
    Takım (dataframe) + gün bazlı fikstürden:
    - Günlük maç sayısı
    - Günlük ve haftalık tahmini FP ve temel kategoriler
    Sadece sağlıklı oyuncular (🔴 Sakat hariç) kullanılır.
    Eğer ESPN fikstürü boş dönerse, Yahoo Games_Next_7D üzerinden fallback dağılımı yapılır.
    """
    proj_cats = ['3PTM','PTS','REB','AST','ST','BLK','TO']

    # Sakat olmayan oyuncular
    active = team_df[~team_df['Health'].astype(str).str.contains("Sakat", na=False)].copy()
    if active.empty:
        today = datetime.now().date()
        days = [today + timedelta(days=i) for i in range(7)]
        daily_rows = [{
            'Tarih': d.strftime('%d.%m.%Y'),
            'Maç_Sayısı_Sen': 0,
            'Proj_FP_Sen': 0.0,
            **{f"{c}_Sen": 0.0 for c in proj_cats}
        } for d in days]
        weekly = {
            'games': 0,
            'fp': 0.0,
            'cats': {c: 0.0 for c in proj_cats}
        }
        return daily_rows, weekly

    today = datetime.now().date()
    days = [today + timedelta(days=i) for i in range(7)]
    day_keys = [d.strftime('%Y-%m-%d') for d in days]

    daily_rows = []
    # 1) ESPN'den gerçek fikstürü kullanmayı dene
    for d_obj, d_key in zip(days, day_keys):
        team_sched = sched_by_day.get(d_key, {}) if sched_by_day else {}
        games_today = 0
        cat_vals = {c: 0.0 for c in proj_cats}
        fp = 0.0

        for _, row in active.iterrows():
            tm = row['Real_Team']
            g_cnt = team_sched.get(tm, 0)
            if g_cnt <= 0:
                continue
            games_today += g_cnt
            for c in proj_cats:
                cat_vals[c] += float(row[c]) * g_cnt
            fp += float(row['Skor']) * g_cnt

        daily_rows.append({
            'Tarih': d_obj.strftime('%d.%m.%Y'),
            'Maç_Sayısı_Sen': int(games_today),
            'Proj_FP_Sen': float(fp),
            **{f"{c}_Sen": float(cat_vals[c]) for c in proj_cats}
        })

    week_games = sum(r['Maç_Sayısı_Sen'] for r in daily_rows)

    # 2) Eğer ESPN verisi 0 maç döndürdüyse, Yahoo Games_Next_7D üzerinden fallback
    if week_games == 0:
        # Oyuncu başına tahmini maç sayıları (Games_Next_7D)
        if 'Games_Next_7D' not in active.columns:
            # yine hiç veri yoksa tamamen sıfırla dön
            weekly = {
                'games': 0,
                'fp': 0.0,
                'cats': {c: 0.0 for c in proj_cats}
            }
            # daily_rows zaten 0'lı
            return daily_rows, weekly

        total_games = int(active['Games_Next_7D'].sum())
        if total_games == 0:
            weekly = {
                'games': 0,
                'fp': 0.0,
                'cats': {c: 0.0 for c in proj_cats}
            }
            return daily_rows, weekly

        # Haftalık toplam kategori & FP (oyuncu per-game * Games_Next_7D)
        week_cats = {
            c: float((active[c] * active['Games_Next_7D']).sum())
            for c in proj_cats
        }
        week_fp = float((active['Skor'] * active['Games_Next_7D']).sum())

        # Toplam maç sayısını 7 güne dağıt
        base = total_games // 7
        extra = total_games % 7
        games_dist = [base + (i < extra) for i in range(7)]

        daily_rows = []
        for d_obj, g_day in zip(days, games_dist):
            frac = g_day / total_games if total_games > 0 else 0.0
            row = {
                'Tarih': d_obj.strftime('%d.%m.%Y'),
                'Maç_Sayısı_Sen': int(g_day),
                'Proj_FP_Sen': float(week_fp * frac)
            }
            for c in proj_cats:
                row[f"{c}_Sen"] = float(week_cats[c] * frac)
            daily_rows.append(row)

        weekly = {
            'games': int(total_games),
            'fp': float(week_fp),
            'cats': week_cats
        }
        return daily_rows, weekly

    # 3) ESPN verisi çalıştıysa, oradan haftalık toplamları çıkar
    week_fp = sum(r['Proj_FP_Sen'] for r in daily_rows)
    week_cats = {
        c: sum(r[f"{c}_Sen"] for r in daily_rows) for c in proj_cats
    }
    weekly = {
        'games': int(week_games),
        'fp': float(week_fp),
        'cats': week_cats
    }
    return daily_rows, weekly

# ==========================================
# APP UI
# ==========================================
st.title("🏀 Burak's GM Dashboard v15.0")

with st.sidebar:
    if st.button("Yenile"):
        st.cache_data.clear()
        st.rerun()
    hide_inj = st.checkbox("Sakatları gizle")
    punt = st.multiselect("Punt Kategorileri", ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO'])

df, lg = load_data(DATA_VERSION)

if df is not None and not df.empty:
    df['Team'] = df['Team'].astype(str).str.strip()
    
    df, act = get_z_and_trade_val(df, punt)
    weak, strong = analyze_needs(df, MY_TEAM_NAME, act)
    
    # 🔧 Sakat + Riskli oyuncuları gizle (Kadro & Takas görünümü için)
    if hide_inj:
        inj_mask = df['Health'].astype(str).str.contains("Sakat|Riskli", regex=True, na=False)
        v_df = df.loc[~inj_mask].copy()
    else:
        v_df = df.copy()
    
    c1, c2 = st.columns(2)
    c1.error(f"Hedeflenmesi Gereken Kategoriler: {', '.join(weak)}")
    c2.success(f"Güçlü Olduğun Kategoriler: {', '.join(strong)}")
    
    t1, t2, t3 = st.tabs(["Kadro Analizi", "Takas Sihirbazı", "Rakip Analizi"])
    
    # -------------------------- Kadro --------------------------
    with t1:
        tm = st.selectbox(
            "Takım Seç",
            [MY_TEAM_NAME] + sorted([t for t in df['Team'].unique() if t != MY_TEAM_NAME])
        )
        show = v_df[v_df['Team'] == tm].sort_values('Skor', ascending=False)
        st.dataframe(
            show[['Player','Pos','Games_Next_7D','Trend','Health','Skor','GP','MPG',
                  'FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']], 
            column_config={
                "Skor": st.column_config.ProgressColumn(
                    "Verimlilik Puanı", format="%.1f", min_value=0, max_value=60
                ),
                "Trend": st.column_config.TextColumn("Form Durumu"),
                "MPG": st.column_config.NumberColumn("Dakika", format="%.1f")
            },
            use_container_width=True, 
            hide_index=True
        )
    
    # -------------------------- Takas Sihirbazı --------------------------
    with t2:
        st.subheader("Takas Sihirbazı – Profesyonel Değerlendirme")
        st.caption("Takas paketlerini, takım ihtiyaçlarını, kategori etkilerini ve sakatlık risklerini birlikte değerlendirir.")
        
        ops = sorted([t for t in df['Team'].unique() if t != MY_TEAM_NAME and t != "Free Agent"])
        col_filters = st.columns(2)
        with col_filters[0]:
            op = st.selectbox("Hedef Takım Seç", ops)
        with col_filters[1]:
            min_score = st.slider("Minimum Uygunluk Puanı", 0.0, 25.0, 5.0, 0.5)
        show_inj_trades = st.checkbox("Sakat oyuncu içeren senaryoları göster", value=False)
        
        if st.button("Takas Senaryolarını Hesapla"):
            res = trade_engine_grouped(df, MY_TEAM_NAME, op, weak)
            
            all_scenarios = []
            for g_name, d in res.items():
                if not d.empty:
                    temp = d.copy()
                    temp["Paket_Tipi"] = g_name
                    all_scenarios.append(temp)
            if all_scenarios:
                all_scenarios = pd.concat(all_scenarios, ignore_index=True)
                all_scenarios_f = all_scenarios[
                    all_scenarios["Uygunluk_Puanı"] >= min_score
                ].copy()
                if not show_inj_trades:
                    all_scenarios_f = all_scenarios_f[all_scenarios_f["Durum"] != "⚠️ RİSKLİ"]
                top5 = all_scenarios_f.sort_values(by="Uygunluk_Puanı", ascending=False).head(5)
                
                st.markdown("### 🔍 En Güçlü 5 Takas Senaryosu (Özet)")
                if not top5.empty:
                    st.dataframe(
                        top5[["Paket_Tipi","Senaryo","Verilecekler","Alınacaklar",
                              "Uygunluk_Puanı","Kategori_Etkisi","Durum","Şans"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Filtrelere uyan güçlü senaryo bulunamadı.")
            
            st.markdown("### 📦 Paket Tiplerine Göre Detaylı Senaryolar")
            ts = st.tabs(list(res.keys()))
            for t_tab, (k, d) in zip(ts, res.items()):
                with t_tab:
                    if not d.empty:
                        df_f = d[d["Uygunluk_Puanı"] >= min_score].copy()
                        if not show_inj_trades:
                            df_f = df_f[df_f["Durum"] != "⚠️ RİSKLİ"]
                        if df_f.empty:
                            st.info("Bu paket tipi için filtrelere uyan senaryo yok.")
                        else:
                            st.dataframe(
                                df_f[['Senaryo','Verilecekler','Alınacaklar',
                                      'Uygunluk_Puanı','Kategori_Etkisi','Durum','Şans']].head(20),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Uygunluk_Puanı": st.column_config.NumberColumn(
                                        "Takıma Uygunluk Puanı", format="%.1f"
                                    ),
                                    "Kategori_Etkisi": st.column_config.TextColumn(
                                        "Etkilenen Kategoriler (Δ z-score)"
                                    ),
                                    "Durum": st.column_config.TextColumn("Risk"),
                                    "Şans": st.column_config.TextColumn("Kabul Edilme Olasılığı")
                                }
                            )
                    else:
                        st.info("Bu paket tipinde mantıklı takas senaryosu bulunamadı.")
    
    # -------------------------- Rakip Analizi --------------------------
    with t3:
        st.subheader("Rakip Karşılaştırma – Sezon, Haftalık Projeksiyon ve Eşleşme Analizi")
        ops = sorted([t for t in df['Team'].unique() if t != MY_TEAM_NAME and t != "Free Agent"])
        op_a = st.selectbox("Rakip Takım Seç", ops)
        
        if op_a:
            cats = ['FG%','FT%','3PTM','PTS','REB','AST','ST','BLK','TO']
            
            my_team_df = df[df['Team'] == MY_TEAM_NAME].copy()
            opp_team_df = df[df['Team'] == op_a].copy()
            
            my_season = my_team_df[cats].mean()
            opp_season = opp_team_df[cats].mean()

            tab_season, tab_weekly, tab_matchup = st.tabs([
                "Sezon Ortalamaları",
                "Haftalık Projeksiyon (Takvim + Sezon Ortalaması)",
                "Haftalık Eşleşme Analizi"
            ])

            # ---- Sezon Ortalamaları ----
            with tab_season:
                data = []
                sm, so = 0, 0
                for c in cats:
                    w = (my_season[c] < opp_season[c]) if c == 'TO' else (my_season[c] > opp_season[c])
                    if w:
                        sm += 1 
                    else:
                        so += 1
                    data.append({
                        'Kategori': c,
                        'Sen (Sezon)': f"{my_season[c]:.1f}",
                        'Rakip (Sezon)': f"{opp_season[c]:.1f}",
                        'Durum': "✅ Üstünsün" if w else "❌ Geri"
                    })
                c1, c2 = st.columns(2)
                c1.metric("Kategori Skoru (Sezon)", f"{sm} - {so}")
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            # ---- Haftalık Projeksiyon ----
            with tab_weekly:
                proj_cats = ['3PTM','PTS','REB','AST','ST','BLK','TO']
                my_proj = {}
                opp_proj = {}
                
                for c in proj_cats:
                    my_proj[c] = float((my_team_df[c] * my_team_df['Games_Next_7D']).sum())
                    opp_proj[c] = float((opp_team_df[c] * opp_team_df['Games_Next_7D']).sum())
                
                data = []
                sm, so = 0, 0
                for c in cats:
                    if c in proj_cats:
                        my_val = my_proj[c]
                        opp_val = opp_proj[c]
                        w = (my_val < opp_val) if c == 'TO' else (my_val > opp_val)
                    else:
                        my_val = my_season[c]
                        opp_val = opp_season[c]
                        w = (my_val < opp_val) if c == 'TO' else (my_val > opp_val)
                    if w:
                        sm += 1
                    else:
                        so += 1
                    data.append({
                        'Kategori': c,
                        'Sen (Hafta)': f"{my_val:.1f}",
                        'Rakip (Hafta)': f"{opp_val:.1f}",
                        'Not': "Sezon ortalaması" if c in ['FG%','FT%'] else "Haftalık projeksiyon",
                        'Durum': "✅ Üstünsün" if w else "❌ Geri"
                    })
                
                my_week_games = int(my_team_df['Games_Next_7D'].sum())
                opp_week_games = int(opp_team_df['Games_Next_7D'].sum())
                my_week_fp = float((my_team_df['Skor'] * my_team_df['Games_Next_7D']).sum())
                opp_week_fp = float((opp_team_df['Skor'] * opp_team_df['Games_Next_7D']).sum())
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Kategori Skoru (Hafta)", f"{sm} - {so}")
                c2.metric("Bu Hafta Maç Sayısı (Sen / Rakip)", f"{my_week_games} / {opp_week_games}")
                c3.metric("Haftalık Fantezi Puanı Projeksiyonu", f"{my_week_fp:.0f} / {opp_week_fp:.0f}")
                
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            # ---- Haftalık Eşleşme Analizi ----
            with tab_matchup:
                st.caption("Bu sekmede, seçtiğin rakibe karşı önümüzdeki 7 gün için maç sayıları ve tahmini üretim; 🔴 Sakat oyuncuların maçları düşülerek hesaplanır (ESPN yoksa Yahoo takvim fallback).")

                sched_by_day = get_schedule_espn_by_day()

                my_daily_rows, my_weekly = project_team_week(my_team_df, sched_by_day)
                opp_daily_rows, opp_weekly = project_team_week(opp_team_df, sched_by_day)

                proj_cats = ['3PTM','PTS','REB','AST','ST','BLK','TO']
                rows = []
                for my_row, opp_row in zip(my_daily_rows, opp_daily_rows):
                    row = {
                        'Tarih': my_row['Tarih'],
                        'Sen Maç': my_row['Maç_Sayısı_Sen'],
                        'Rakip Maç': opp_row['Maç_Sayısı_Sen'],
                        'Sen FP': my_row['Proj_FP_Sen'],
                        'Rakip FP': opp_row['Proj_FP_Sen'],
                    }
                    for c in proj_cats:
                        row[f"{c} Sen"] = my_row.get(f"{c}_Sen", 0.0)
                        row[f"{c} Rakip"] = opp_row.get(f"{c}_Sen", 0.0)
                    rows.append(row)

                daily_df = pd.DataFrame(rows)

                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam Potansiyel Maç (Sağlıklı Oyuncular)", f"{my_weekly['games']} / {opp_weekly['games']}")
                c2.metric("Haftalık FP Projeksiyonu (Sağlıklı Oyuncular)", f"{my_weekly['fp']:.0f} / {opp_weekly['fp']:.0f}")
                
                diff_win = 0
                diff_lose = 0
                cat_rows = []
                for c in proj_cats:
                    my_val = my_weekly['cats'][c]
                    opp_val = opp_weekly['cats'][c]
                    w = (my_val < opp_val) if c == 'TO' else (my_val > opp_val)
                    if w:
                        diff_win += 1
                    else:
                        diff_lose += 1
                    cat_rows.append({
                        "Kategori": c,
                        "Sen (Hafta)": f"{my_val:.1f}",
                        "Rakip (Hafta)": f"{opp_val:.1f}",
                        "Durum": "✅ Üstünsün" if w else "❌ Geri"
                    })

                c3.metric("Haftalık Eşleşme Kategori Skoru", f"{diff_win} - {diff_lose}")

                st.markdown("#### Günlük Maç & FP Projeksiyonu (Sakatlar Hariç)")
                st.dataframe(daily_df, use_container_width=True, hide_index=True)

                st.markdown("#### Kategori Bazlı Haftalık Projeksiyon (Sakatlar Hariç)")
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
