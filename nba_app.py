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
from datetime import datetime, timedelta

# --- NBA API ---
from nba_api.stats.endpoints import leaguedashplayerstats, scoreboardv2

# --- AYARLAR ---
SEASON_YEAR = 2025
NBA_SEASON_STRING = '2025-26' 
TARGET_LEAGUE_ID = "61142" 
MY_TEAM_NAME = "Burak's Wizards"
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- YAN PANEL ---
with st.sidebar:
    st.header("GM Kontrol Paneli")
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("🎯 Strateji Ayarları")
    punt_cats = st.multiselect(
        "Gözden Çıkar (Punt):", 
        ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO'],
        help="Bu kategoriler hesaplamada 'yok' sayılır."
    )
    st.markdown("---")
    st.info("Kapsam: Yahoo Puanlar + NBA Dakikalar + Fikstür + Takas + Rakip Analizi")

# ==========================================
# 1. VERİ ÇEKME FONKSİYONLARI (NBA & YAHOO)
# ==========================================

@st.cache_data(ttl=3600)
def get_nba_real_stats():
    """NBA.com'dan gerçek GP ve MPG verilerini çeker"""
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=NBA_SEASON_STRING, per_mode_detailed='PerGame')
        df = stats.get_data_frames()[0]
        nba_data = {}
        for index, row in df.iterrows():
            clean_name = row['PLAYER_NAME'].lower().replace('.', '').strip()
            nba_data[clean_name] = {'GP': row['GP'], 'MPG': row['MIN'], 'TEAM': row['TEAM_ABBREVIATION']}
        return nba_data
    except Exception: return {}

@st.cache_data(ttl=3600)
def get_nba_schedule_next_7_days():
    """Önümüzdeki 7 günün fikstürünü çeker ve takım başına maç sayısını hesaplar"""
    try:
        team_game_counts = {}
        today = datetime.now()
        
        # Önümüzdeki 7 günü tara
        for i in range(7):
            date_str = (today + timedelta(days=i)).strftime('%m/%d/%Y')
            # Scoreboard o günkü maçları getirir
            board = scoreboardv2.ScoreboardV2(game_date=date_str)
            games = board.game_header.get_data_frame()
            
            # O gün oynayan takımları listeye ekle
            for _, game in games.iterrows():
                home = game['HOME_TEAM_ID'] # ID döner, maplemek gerekebilir ama basitçe sayabiliriz
                visitor = game['VISITOR_TEAM_ID']
                
                # Takım ID'lerini isme çevirmek için LineScore kullanabiliriz ama 
                # NBA API bazen yavaşlar. Basitlik için Yahoo Team Abbr kullanacağız.
                # Burada Scoreboard'dan takım kısaltmalarını almak için 'line_score' dataframe'ine bakılır.
            
            line_score = board.line_score.get_data_frame()
            playing_teams = line_score['TEAM_ABBREVIATION'].unique()
            
            for team in playing_teams:
                team_game_counts[team] = team_game_counts.get(team, 0) + 1
                
        return team_game_counts
    except Exception as e:
        print(f"Fikstür hatası: {e}")
        return {}

@st.cache_data(ttl=3600)
def load_data():
    # 1. NBA Verilerini Hazırla (İstatistik + Fikstür)
    nba_stats_dict = get_nba_real_stats()
    nba_schedule = get_nba_schedule_next_7_days()
    
    # 2. Secrets Kontrolü
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except: return None, None
        else: return None, None

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
        
        if not target_league_key: return None, None

        lg = gm.to_league(target_league_key)
        
        all_data = []
        teams = lg.teams()
        
        progress_bar = st.progress(0, text="Sezon verileri ve Fikstür işleniyor...")
        
        # 1. TAKIMLARI TARA
        total_teams = len(teams)
        step = 0
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    # Çift Sorgu (Sezon + Son 1 Ay Trendi)
                    stats_season = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    stats_month = lg.player_stats(p_ids, 'last_month')
                    
                    for i, pm in enumerate(roster):
                        process_player_full(pm, stats_season[i], stats_month[i], t_name, "Sahipli", all_data, nba_stats_dict, nba_schedule)
            except: pass
            step += 1
            progress_bar.progress(step / (total_teams + 1))

        # 2. FREE AGENT TARA (TOP 300)
        try:
            progress_bar.progress(0.90, text="🆓 300 Free Agent + Fikstür taranıyor...")
            fa_players = lg.free_agents(None)[:300]
            fa_ids = [p['player_id'] for p in fa_players]
            
            chunk_size = 25
            for i in range(0, len(fa_ids), chunk_size):
                chunk_ids = fa_ids[i:i + chunk_size]
                chunk_p = fa_players[i:i + chunk_size]
                try:
                    stats_s = lg.player_stats(chunk_ids, ANALYSIS_TYPE)
                    stats_m = lg.player_stats(chunk_ids, 'last_month')
                    
                    for k, pm in enumerate(chunk_p):
                        process_player_full(pm, stats_s[k], stats_m[k], "🆓 FA", "Free Agent", all_data, nba_stats_dict, nba_schedule)
                except: pass
                time.sleep(0.1)
        except: pass

        progress_bar.empty()
        return pd.DataFrame(all_data) if all_data else None, lg
        
    except Exception: return None, None

def process_player_full(meta, stat_s, stat_m, team_name, ownership, data_list, nba_dict, nba_schedule):
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        p_name = meta['name']
        
        # Pozisyon Sadeleştirme
        raw_pos = meta.get('display_position', '')
        simple_pos = raw_pos.replace('PG', 'G').replace('SG', 'G').replace('SF', 'F').replace('PF', 'F')
        u_pos = list(set(simple_pos.split(',')))
        u_pos.sort(key=lambda x: 1 if x=='G' else (2 if x=='F' else 3))
        final_pos = ",".join(u_pos)

        # NBA Verisi Eşleştirme (GP, MPG ve Takım Kısaltması)
        c_name = p_name.lower().replace('.', '').strip()
        real_gp = 0
        real_mpg = 0.0
        nba_team_abbr = "N/A"
        
        if c_name in nba_dict:
            real_gp = nba_dict[c_name]['GP']
            real_mpg = nba_dict[c_name]['MPG']
            nba_team_abbr = nba_dict[c_name]['TEAM']
        else:
            # İsim eşleşmezse Yahoo'dan gelen takım kısaltmasını kullanmaya çalış
            nba_team_abbr = meta.get('editorial_team_abbr', 'N/A').upper()

        # Fikstür Avantajı (Gelecek 7 Gün Maç Sayısı)
        # NBA API takımları 'GSW', 'LAL' gibi kısaltır. Yahoo da benzer.
        # Ufak düzeltmeler gerekebilir (PHO -> PHX gibi) ama genelde tutar.
        if nba_team_abbr == 'PHO': nba_team_abbr = 'PHX' 
        next_games = nba_schedule.get(nba_team_abbr, 0)
        
        # Sakatlık
        st_code = meta.get('status', '')
        if st_code in ['INJ', 'O']: inj = f"🟥 {st_code}"
        elif st_code in ['GTD', 'DTD']: inj = f"Rx {st_code}"
        else: inj = "✅"

        # Trend Hesabı
        pts_s = get_val(stat_s.get('PTS'))
        pts_m = get_val(stat_m.get('PTS'))
        trend_score = pts_m - pts_s if pts_m > 0 else 0
        
        if trend_score > 3.0: trend_icon = "🔥"
        elif trend_score > 1.0: trend_icon = "↗️"
        elif trend_score < -3.0: trend_icon = "🥶"
        elif trend_score < -1.0: trend_icon = "↘️"
        else: trend_icon = "➖"

        # Yüzdeler
        fg_val = get_val(stat_s.get('FG%')) * 100
        ft_val = get_val(stat_s.get('FT%')) * 100

        data_list.append({
            'Player': p_name,
            'Team': team_name,
            'Real_Team': nba_team_abbr, # NBA Takımı (Fikstür için)
            'Owner_Status': ownership,
            'Pos': final_pos,
            'Health': inj,
            'Trend': trend_icon,
            'Games_Next_7D': int(next_games), # YENİ VERİ: Gelecek 7 gün maç sayısı
            'GP': int(real_gp),
            'MPG': float(real_mpg),
            'FG%': fg_val, 
            'FT%': ft_val,
            '3PTM': get_val(stat_s.get('3PTM')),
            'PTS': pts_s,
            'REB': get_val(stat_s.get('REB')),
            'AST': get_val(stat_s.get('AST')),
            'ST': get_val(stat_s.get('ST')),
            'BLK': get_val(stat_s.get('BLK')),
            'TO': get_val(stat_s.get('TO'))
        })
    except: pass

# ==========================================
# 2. ANALİZ & HESAPLAMA MOTORU
# ==========================================

def calculate_z_scores(df, punt_list):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    active_cats = [c for c in cats if c not in punt_list]
    
    if df.empty: return df, active_cats
    
    for c in cats:
        if c in punt_list:
            df[f'z_{c}'] = 0.0
            continue
            
        if c not in df.columns: df[c] = 0.0
        mean, std = df[c].mean(), df[c].std()
        if std == 0: std = 1
        df[f'z_{c}'] = (mean - df[c]) / std if c == 'TO' else (df[c] - mean) / std
    
    # Genel Kalite (Punt hariç toplam değer)
    df['Genel_Kalite'] = df[[f'z_{c}' for c in active_cats]].sum(axis=1)
    return df, active_cats

def analyze_needs(df, my_team, active_cats):
    z_cols = [f'z_{c}' for c in active_cats]
    m_df = df[df['Team'] == my_team]
    if m_df.empty: return [], []
    prof = m_df[z_cols].sum().sort_values()
    return [x.replace('z_', '') for x in prof.head(4).index], [x.replace('z_', '') for x in prof.tail(3).index]

def score_players(df, targets, active_cats):
    df['Skor'] = 0
    for c in active_cats:
        if f'z_{c}' in df.columns:
            w = 3.0 if c in targets else 1.0
            df['Skor'] += df[f'z_{c}'] * w
            
    # FİKSTÜR AVANTAJI PUANI EKLEME
    # Eğer oyuncu önümüzdeki hafta 4 maç yapıyorsa skorunu artır (Streaming için)
    # 4 Maç: +1.5 Puan, 2 Maç: -1.0 Puan
    # Bu sadece Free Agent seçerken çok kritiktir.
    df['Fikstür_Etkisi'] = df['Games_Next_7D'].apply(lambda x: 1.5 if x >= 4 else (-1.0 if x <= 2 else 0))
    # Ana skora ufak bir etki ekleyelim (Tercihe bağlı, şimdilik sadece görsel kalsın)
    # df['Skor'] += df['Fikstür_Etkisi'] 
    
    return df

# --- RAKİP ANALİZ MODÜLÜ ---
def analyze_weekly_matchup(lg, df, my_team):
    try:
        teams_list = df[df['Owner_Status'] == 'Sahipli']['Team'].unique()
        teams_list = [t for t in teams_list if t != my_team]
        return teams_list
    except: return []

def get_matchup_prediction(df, my_team, opp_team):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    my_stats = df[df['Team'] == my_team][cats].mean()
    opp_stats = df[df['Team'] == opp_team][cats].mean()
    
    score_me, score_opp = 0, 0
    details = []
    
    for c in cats:
        val_me, val_opp = my_stats[c], opp_stats[c]
        win = (val_me < val_opp) if c == 'TO' else (val_me > val_opp)
        
        if win: score_me += 1
        else: score_opp += 1
            
        details.append({
            'Kategori': c, 'Benim Ort': f"{val_me:.1f}", 'Rakip Ort': f"{val_opp:.1f}",
            'Durum': "✅" if win else "❌"
        })
    return score_me, score_opp, pd.DataFrame(details)

# --- ULTIMATE TAKAS MOTORU (1v1 -> 3v3) ---
def ultimate_trade_engine(df, my_team, targets):
    my_assets = df[df['Team'] == my_team].sort_values(by='Skor', ascending=True).head(7)
    opponents = df[(df['Team'] != my_team) & (df['Owner_Status'] == 'Sahipli')]['Team'].unique()
    proposals = []
    prog_bar = st.progress(0, text="Milyarlarca olasılık hesaplanıyor...")
    
    for idx, opp_team in enumerate(opponents):
        opp_assets = df[df['Team'] == opp_team].sort_values(by='Skor', ascending=False).head(6)
        for n_give in range(1, 4): 
            for n_recv in range(1, 4):
                my_combos = list(itertools.combinations(my_assets.index, n_give))
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
    total_give_val = sum([p['Genel_Kalite'] for p in give_list])
    total_recv_val = sum([p['Genel_Kalite'] for p in recv_list])
    val_diff = total_give_val - total_recv_val
    
    if val_diff > -4.0:
        total_give_score = sum([p['Skor'] for p in give_list])
        total_recv_score = sum([p['Skor'] for p in recv_list])
        gain = total_recv_score - total_give_score
        threshold = 2.0 + (len(give_list) + len(recv_list)) * 0.5
        
        if gain > threshold:
            give_names = ", ".join([p['Player'] for p in give_list])
            recv_names = ", ".join([p['Player'] for p in recv_list])
            impact_text = get_package_impact_text(give_list, recv_list)
            proposal_list.append({
                'Tür': f"{len(give_list)}v{len(recv_list)}",
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
        recv_tot = sum([p[c] for p in r_list])
        diff = recv_tot - give_tot
        if diff > 1.5: improvements.append(f"{c} (+{diff:.1f})")
    return f"🚀 {', '.join(improvements)}" if improvements else "Genel İyileşme"

# ==========================================
# 3. ANA ARAYÜZ (DASHBOARD)
# ==========================================
st.title("🏀 Burak's Ultimate GM Dashboard")
st.markdown("---")

df, lg = load_data()

if df is not None and not df.empty:
    df, active_cats = calculate_z_scores(df, punt_cats)
    targets, strengths = analyze_needs(df, MY_TEAM_NAME, active_cats)
    
    if targets:
        df = score_players(df, targets, active_cats)
        
        # Üst Bilgi
        c1, c2, c3 = st.columns(3)
        c1.error(f"📉 Zayıf: {', '.join(targets)}")
        c2.success(f"📈 Güçlü: {', '.join(strengths)}")
        if punt_cats: c3.warning(f"🚫 Punt: {', '.join(punt_cats)}")
        else: c3.info("🚫 Punt Yok")
        
        st.markdown("---")
        
        # Filtreler
        col1, col2 = st.columns(2)
        f_stat = col1.multiselect("Filtre:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        h_inj = col2.checkbox("Sadece Sağlamlar (✅)", value=False)
        
        v_df = df.copy()
        if f_stat: v_df = v_df[v_df['Owner_Status'].isin(f_stat)]
        if h_inj: v_df = v_df[v_df['Health'].str.contains("✅")]

        # SÜTUN LİSTESİ (Fikstür Eklendi: Games_Next_7D)
        all_cols = ['Player', 'Team', 'Real_Team', 'Games_Next_7D', 'Trend', 'Pos', 'Health', 'GP', 'MPG', 'Skor', 
                    'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Hedefler & Fikstür", "📋 Kadrom", "🌍 Tüm Liste", "🧙‍♂️ Takas Sihirbazı", "⚔️ Rakip Analizi"])
        
        with tab1:
            st.caption("Gelecek 7 Gün: 4 Maç = 🟢 (Avantaj), 2 Maç = 🔴 (Dezavantaj)")
            trade_df = v_df[v_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            
            st.dataframe(
                trade_df[all_cols].head(50),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Puan", format="%.1f", max_value=trade_df['Skor'].max()),
                    "Games_Next_7D": st.column_config.NumberColumn("7 Günlük Maç", format="%d 🏀"),
                    "Trend": st.column_config.TextColumn("Form"),
                    "Real_Team": st.column_config.TextColumn("NBA"),
                    "FG%": st.column_config.NumberColumn("FG%", format="%.1f"), 
                    "FT%": st.column_config.NumberColumn("FT%", format="%.1f"),
                    "GP": st.column_config.NumberColumn("GP", width="small"),
                    "MPG": st.column_config.NumberColumn("MPG", format="%.1f", width="small"),
                },
                use_container_width=True
            )
            
            st.subheader("💎 Gizli Cevher Haritası")
            fig = px.scatter(
                v_df[v_df['Owner_Status'] == 'Free Agent'], 
                x="MPG", y="Skor", color="Games_Next_7D", hover_name="Player", size="GP",
                title="Free Agentlar: Verim vs Süre vs Fikstür (Sarı renk = Çok Maç)"
            )
            st.plotly_chart(fig)

        with tab2:
            st.dataframe(df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)[all_cols], use_container_width=True)
            
        with tab3:
            st.dataframe(v_df[all_cols], use_container_width=True)
            
        with tab4:
            st.header("🧙‍♂️ Ultimate Takas Motoru")
            if st.button("🚀 Olası Senaryoları Hesapla (3v3 Dahil)"):
                prop_df = ultimate_trade_engine(df, MY_TEAM_NAME, targets)
                if not prop_df.empty:
                    st.dataframe(prop_df.head(50), column_config={"Adalet": st.column_config.ProgressColumn("Şans", min_value=-5, max_value=5)}, use_container_width=True)
                else: st.warning("Takas bulunamadı.")
        
        with tab5:
            st.header("⚔️ Rakip Analizi")
            opponents_list = analyze_weekly_matchup(lg, df, MY_TEAM_NAME)
            if opponents_list:
                sel_opp = st.selectbox("Rakip Seç:", opponents_list)
                if sel_opp:
                    s1, s2, d_df = get_matchup_prediction(df, MY_TEAM_NAME, sel_opp)
                    c_m1, c_m2 = st.columns(2)
                    c_m1.metric("Sen", s1)
                    c_m2.metric(sel_opp, s2)
                    st.dataframe(d_df, use_container_width=True)

else:
    st.info("Sistem başlatılıyor (Fikstür ve İstatistikler çekiliyor)...")
