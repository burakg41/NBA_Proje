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
# Timeout hatalarını önlemek için header ekleyelim (Opsiyonel ama önerilir)
from nba_api.stats.endpoints import leaguedashplayerstats, scoreboardv2

# --- AYARLAR ---
SEASON_YEAR = 2025  # Yahoo'da 2025-2026 sezonu "2025" olarak geçer.
NBA_SEASON_STRING = '2025-26' 
TARGET_LEAGUE_ID = "61142"  # Senin Lig ID'n
MY_TEAM_NAME = "Burak's Wizards" # Kendi takım ismin (Tam eşleşmeli)
ANALYSIS_TYPE = 'average_season' 

# Yahoo ve NBA Takım Kısaltmaları Eşleştirme Haritası (Hataları önler)
YAHOO_TO_NBA_MAP = {
    'NY': 'NYK', 'GS': 'GSW', 'NO': 'NOP', 'SA': 'SAS', 'PHO': 'PHX', 
    'WAS': 'WAS', 'UTAH': 'UTA', 'CHA': 'CHA', 'BKN': 'BKN'
}

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide", page_icon="🏀")

# --- YAN PANEL ---
with st.sidebar:
    st.header("🏀 GM Kontrol Paneli")
    if st.button("🔄 Verileri Yenile", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("🎯 Strateji Ayarları")
    punt_cats = st.multiselect(
        "Gözden Çıkar (Punt):", 
        ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO'],
        help="Bu kategoriler hesaplamada 'yok' sayılır."
    )
    st.info("ℹ️ Not: NBA API bazen yavaş yanıt verebilir, lütfen bekleyiniz.")

# ==========================================
# 1. VERİ ÇEKME FONKSİYONLARI (NBA & YAHOO)
# ==========================================

@st.cache_data(ttl=3600)
def get_nba_real_stats():
    """NBA.com'dan gerçek GP ve MPG verilerini çeker"""
    try:
        # Headers eklemek bazen bloklanmayı engeller
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=NBA_SEASON_STRING, 
            per_mode_detailed='PerGame',
            timeout=30
        )
        df = stats.get_data_frames()[0]
        nba_data = {}
        for index, row in df.iterrows():
            # İsim temizliği: "Luka Dončić" -> "lukadoncic" gibi
            clean_name = row['PLAYER_NAME'].lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
            nba_data[clean_name] = {'GP': row['GP'], 'MPG': row['MIN'], 'TEAM': row['TEAM_ABBREVIATION']}
        return nba_data
    except Exception as e: 
        st.warning(f"⚠️ NBA İstatistikleri çekilemedi: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_nba_schedule_next_7_days():
    """Önümüzdeki 7 günün fikstürünü çeker"""
    try:
        team_game_counts = {}
        today = datetime.now()
        
        for i in range(7):
            date_str = (today + timedelta(days=i)).strftime('%m/%d/%Y')
            try:
                board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=10)
                line_score = board.line_score.get_data_frame()
                
                if not line_score.empty:
                    playing_teams = line_score['TEAM_ABBREVIATION'].unique()
                    for team in playing_teams:
                        team_game_counts[team] = team_game_counts.get(team, 0) + 1
            except:
                continue # Tek bir gün hata verirse atla
            time.sleep(0.3) # API'yi boğmamak için minik bekleme
                
        return team_game_counts
    except Exception as e:
        st.warning(f"⚠️ Fikstür verisi çekilemedi: {e}")
        return {}

def authenticate_yahoo():
    """Yahoo Auth işlemlerini yönetir"""
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                # Sayısal değerleri string hatasına karşı float/int yapalım
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except Exception as e:
                st.error(f"❌ Secrets dosyasından json oluşturulamadı: {e}")
                return None
        else:
            st.error("❌ 'oauth2.json' bulunamadı ve Secrets ayarlanmamış!")
            return None

    try:
        sc = OAuth2(None, None, from_file='oauth2.json')
        if not sc.token_is_valid():
            sc.refresh_access_token()
        return sc
    except Exception as e:
        st.error(f"❌ Yahoo Giriş Hatası: {e}")
        return None

@st.cache_data(ttl=3600)
def load_data():
    # 1. NBA Verilerini Hazırla (Paralel gibi çalışsın diye başta çağırıyoruz)
    with st.spinner('NBA Verileri (Gerçek İstatistikler + Fikstür) çekiliyor...'):
        nba_stats_dict = get_nba_real_stats()
        nba_schedule = get_nba_schedule_next_7_days()
    
    # 2. Yahoo Bağlantısı
    sc = authenticate_yahoo()
    if not sc: return None, None

    try:
        gm = yfa.Game(sc, 'nba')
        
        # Doğru Ligi Bulma
        league_ids = gm.league_ids(year=SEASON_YEAR)
        target_league_key = None
        
        # ID listesinde string eşleşmesi yap
        for lid in league_ids:
            if str(TARGET_LEAGUE_ID) in lid:
                target_league_key = lid
                break
        
        if not target_league_key:
            st.error(f"❌ Lig ID ({TARGET_LEAGUE_ID}) bulunamadı! Mevcut Ligler: {league_ids}")
            return None, None

        lg = gm.to_league(target_league_key)
        teams = lg.teams()
        
        all_data = []
        
        # Progress Bar Ayarı
        progress_text = "Takım verileri indiriliyor..."
        my_bar = st.progress(0, text=progress_text)
        
        # 1. TAKIMLARI TARA
        total_teams = len(teams)
        for idx, team_key in enumerate(teams.keys()):
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                if p_ids:
                    stats_season = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    stats_month = lg.player_stats(p_ids, 'last_month')
                    
                    for i, pm in enumerate(roster):
                        # stats_season[i] bazen boş gelebilir, kontrol et
                        if i < len(stats_season) and i < len(stats_month):
                            process_player_full(pm, stats_season[i], stats_month[i], t_name, "Sahipli", all_data, nba_stats_dict, nba_schedule)
            except Exception as e:
                print(f"Hata ({t_name}): {e}")
            
            # Barı güncelle
            my_bar.progress((idx + 1) / total_teams, text=f"{t_name} işlendi...")

        # 2. FREE AGENT TARA (TOP 200 - Hız İçin Düşürüldü)
        try:
            my_bar.progress(0.95, text="🆓 Free Agent havuzu taranıyor...")
            fa_players = lg.free_agents(None)[:200]
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
                # Yahoo API rate limit yememek için minik bekleme
                time.sleep(0.1)
        except: pass

        my_bar.empty()
        return pd.DataFrame(all_data) if all_data else None, lg
        
    except Exception as e:
        st.error(f"Veri yükleme sırasında genel hata: {e}")
        return None, None

def process_player_full(meta, stat_s, stat_m, team_name, ownership, data_list, nba_dict, nba_schedule):
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            try: return float(val)
            except: return 0.0

        p_name = meta['name']
        
        # Pozisyon Sadeleştirme
        raw_pos = meta.get('display_position', '')
        simple_pos = raw_pos.replace('PG', 'G').replace('SG', 'G').replace('SF', 'F').replace('PF', 'F')
        u_pos = list(set(simple_pos.split(',')))
        u_pos.sort(key=lambda x: 1 if x=='G' else (2 if x=='F' else 3))
        final_pos = ",".join(u_pos)

        # NBA Verisi Eşleştirme
        # İsim temizliği (Nokta, tırnak, tire temizle)
        c_name = p_name.lower().replace('.', '').replace("'", "").replace('-', ' ').strip()
        
        real_gp = 0
        real_mpg = 0.0
        nba_team_abbr = "N/A"
        
        if c_name in nba_dict:
            real_gp = nba_dict[c_name]['GP']
            real_mpg = nba_dict[c_name]['MPG']
            nba_team_abbr = nba_dict[c_name]['TEAM']
        else:
            # İsim eşleşmezse Yahoo'dan gelen takım kısaltmasını kullan
            # Yahoo: 'NY' -> NBA Map -> 'NYK'
            raw_yahoo_team = meta.get('editorial_team_abbr', 'N/A').upper()
            nba_team_abbr = YAHOO_TO_NBA_MAP.get(raw_yahoo_team, raw_yahoo_team)

        # Fikstür Avantajı
        # NBA API'den gelen takım kısaltması ile eşleştir
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
            'Real_Team': nba_team_abbr,
            'Owner_Status': ownership,
            'Pos': final_pos,
            'Health': inj,
            'Trend': trend_icon,
            'Games_Next_7D': int(next_games),
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
    except Exception: pass

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
        mean = df[c].mean()
        std = df[c].std()
        
        if std == 0: std = 1
        
        # TO için düşük olması iyidir (Tersine çevir)
        if c == 'TO':
             df[f'z_{c}'] = (mean - df[c]) / std
        else:
             df[f'z_{c}'] = (df[c] - mean) / std
    
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
    df['Skor'] = 0.0
    for c in active_cats:
        if f'z_{c}' in df.columns:
            w = 3.0 if c in targets else 1.0
            df['Skor'] += df[f'z_{c}'] * w
            
    # Fikstür Etkisi (Opsiyonel: Skoru manipüle edebilir)
    # df['Skor'] += df['Games_Next_7D'] * 0.5 
    return df

# --- RAKİP ANALİZ MODÜLÜ ---
def analyze_weekly_matchup(lg, df, my_team):
    try:
        teams_list = df[df['Owner_Status'] == 'Sahipli']['Team'].unique()
        # Kendini listeden çıkar
        teams_list = [t for t in teams_list if t != my_team]
        return teams_list
    except: return []

def get_matchup_prediction(df, my_team, opp_team):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    
    # Sadece o takımdaki oyuncuları al
    my_df = df[df['Team'] == my_team]
    opp_df = df[df['Team'] == opp_team]
    
    if my_df.empty or opp_df.empty:
        return 0, 0, pd.DataFrame()

    my_stats = my_df[cats].mean()
    opp_stats = opp_df[cats].mean()
    
    score_me, score_opp = 0, 0
    details = []
    
    for c in cats:
        val_me, val_opp = my_stats[c], opp_stats[c]
        # TO için düşük olan kazanır
        if c == 'TO':
            win = val_me < val_opp
        else:
            win = val_me > val_opp
        
        if win: score_me += 1
        elif val_me < val_opp: score_opp += 1
        # Eşitlik durumunda puan eklemiyoruz
            
        details.append({
            'Kategori': c, 
            'Benim Ort': f"{val_me:.1f}", 
            'Rakip Ort': f"{val_opp:.1f}",
            'Durum': "✅" if win else ("🤝" if val_me == val_opp else "❌")
        })
    return score_me, score_opp, pd.DataFrame(details)

# --- ULTIMATE TAKAS MOTORU ---
def ultimate_trade_engine(df, my_team, targets):
    # Hesaplama yükünü azaltmak için limitler
    my_assets = df[df['Team'] == my_team].sort_values(by='Skor', ascending=True).head(7)
    opponents = df[(df['Team'] != my_team) & (df['Owner_Status'] == 'Sahipli')]['Team'].unique()
    
    proposals = []
    prog_bar = st.progress(0, text="Milyarlarca olasılık hesaplanıyor...")
    
    total_ops = len(opponents)
    for idx, opp_team in enumerate(opponents):
        opp_assets = df[df['Team'] == opp_team].sort_values(by='Skor', ascending=False).head(6)
        
        # 1v1, 2v2, 3v3 kombinasyonları
        for n_give in range(1, 4): 
            for n_recv in range(1, 4):
                # Takas oyuncu sayısı dengesi (opsiyonel: n_give == n_recv şartı koyulabilir)
                if abs(n_give - n_recv) > 1: continue 

                my_combos = list(itertools.combinations(my_assets.index, n_give))
                opp_combos = list(itertools.combinations(opp_assets.index, n_recv))
                
                for m_idxs in my_combos:
                    give_list = [df.loc[i] for i in m_idxs]
                    for o_idxs in opp_combos:
                        recv_list = [df.loc[i] for i in o_idxs]
                        analyze_generic_trade(give_list, recv_list, proposals)
        
        prog_bar.progress((idx + 1) / total_ops)
    
    prog_bar.empty()
    return pd.DataFrame(proposals).sort_values(by='Kazanç', ascending=False)

def analyze_generic_trade(give_list, recv_list, proposal_list):
    total_give_val = sum([p['Genel_Kalite'] for p in give_list])
    total_recv_val = sum([p['Genel_Kalite'] for p in recv_list])
    val_diff = total_recv_val - total_give_val # Pozitifse adil/karlı
    
    # Çok dengesiz takasları filtrele (Yahoo "Reject" yememek için)
    if val_diff > -3.0: 
        total_give_score = sum([p['Skor'] for p in give_list])
        total_recv_score = sum([p['Skor'] for p in recv_list])
        gain = total_recv_score - total_give_score
        
        # Sadece belirgin kazanç sağlayanları listele
        threshold = 2.0
        
        if gain > threshold:
            give_names = ", ".join([p['Player'] for p in give_list])
            recv_names = ", ".join([p['Player'] for p in recv_list])
            impact_text = get_package_impact_text(give_list, recv_list)
            
            proposal_list.append({
                'Tür': f"{len(give_list)}v{len(recv_list)}",
                'Verilecekler': give_names,
                'Alınacaklar': recv_names,
                'Hedef Takım': recv_list[0]['Team'],
                'Adalet': round(val_diff, 1),
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
        if diff > 1.0: improvements.append(f"{c}")
    return f"🚀 {', '.join(improvements)}" if improvements else "Genel İyileşme"

# ==========================================
# 3. ANA ARAYÜZ (DASHBOARD)
# ==========================================
st.title("🏀 Burak's Ultimate GM Dashboard")

# Veriyi Yükle
df, lg = load_data()

if df is not None and not df.empty:
    df, active_cats = calculate_z_scores(df, punt_cats)
    targets, strengths = analyze_needs(df, MY_TEAM_NAME, active_cats)
    
    if targets:
        df = score_players(df, targets, active_cats)
        
        # Üst Bilgi
        c1, c2, c3 = st.columns(3)
        c1.error(f"📉 Zayıf Yönler: {', '.join(targets)}")
        c2.success(f"📈 Güçlü Yönler: {', '.join(strengths)}")
        c3.warning(f"🚫 Puntlanan: {', '.join(punt_cats) if punt_cats else 'Yok'}")
        
        st.markdown("---")
        
        # Filtreler
        col1, col2 = st.columns(2)
        f_stat = col1.multiselect("Oyuncu Durumu:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        h_inj = col2.checkbox("Sadece Sağlam Oyuncular (✅)", value=False)
        
        v_df = df.copy()
        if f_stat: v_df = v_df[v_df['Owner_Status'].isin(f_stat)]
        if h_inj: v_df = v_df[v_df['Health'].str.contains("✅")]

        # SÜTUN LİSTESİ
        all_cols = ['Player', 'Team', 'Real_Team', 'Games_Next_7D', 'Trend', 'Pos', 'Health', 'GP', 'MPG', 'Skor', 
                    'FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Hedefler & Fikstür", "📋 Kadrom", "🌍 Tüm Liste", "🧙‍♂️ Takas Sihirbazı", "⚔️ Rakip Analizi"])
        
        with tab1:
            st.caption("Fikstür Avantajı: 4 Maç = Çok İyi, 2 Maç = Kötü. (Streaming için önemlidir)")
            trade_df = v_df[v_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            
            st.dataframe(
                trade_df[all_cols].head(50),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Puan", format="%.1f", max_value=trade_df['Skor'].max()),
                    "Games_Next_7D": st.column_config.NumberColumn("7 Günlük Maç", format="%d 🏀"),
                    "Trend": st.column_config.TextColumn("Form"),
                    "FG%": st.column_config.NumberColumn("FG%", format="%.1f"), 
                    "FT%": st.column_config.NumberColumn("FT%", format="%.1f"),
                },
                use_container_width=True
            )
            
            st.subheader("💎 Gizli Cevher Haritası (Free Agents)")
            fa_only = v_df[v_df['Owner_Status'] == 'Free Agent']
            if not fa_only.empty:
                fig = px.scatter(
                    fa_only, 
                    x="MPG", y="Skor", color="Games_Next_7D", hover_name="Player", size="GP",
                    title="Verim vs Süre vs Fikstür (Sarı renk = Çok Maç)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Grafik için yeterli Free Agent verisi yok.")

        with tab2:
            st.dataframe(df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)[all_cols], use_container_width=True)
            
        with tab3:
            st.dataframe(v_df[all_cols], use_container_width=True)
            
        with tab4:
            st.header("🧙‍♂️ Ultimate Takas Motoru")
            st.info("Bu işlem milyarlarca kombinasyonu tarar. Lütfen sabırlı olun.")
            if st.button("🚀 Olası Senaryoları Hesapla"):
                prop_df = ultimate_trade_engine(df, MY_TEAM_NAME, targets)
                if not prop_df.empty:
                    st.dataframe(
                        prop_df.head(50), 
                        column_config={"Adalet": st.column_config.ProgressColumn("Şans", min_value=-5, max_value=5)}, 
                        use_container_width=True
                    )
                else: st.warning("Uygun takas senaryosu bulunamadı.")
        
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
    st.info("Sistem başlatılıyor... Lütfen bekleyiniz. İlk açılışta verilerin çekilmesi 1-2 dakika sürebilir.")
