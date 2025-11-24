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

# DİKKAT: Artık 'average_season' değil 'season' (Toplamlar) çekiyoruz.
# Ortalamayı kendimiz hesaplayacağız, böylesi çok daha garanti.
ANALYSIS_TYPE = 'season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- YAN PANEL ---
with st.sidebar:
    st.header("Yönetim Paneli")
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()
    st.info(f"📅 Mod: Sezon Toplamları -> Ortalamalar")

# --- VERİ YÜKLEME ---
@st.cache_data(ttl=3600)
def load_data():
    status_container = st.empty()
    
    # --- BULUT İÇİN GİZLİ DOSYA ---
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                secrets_dict = dict(st.secrets['yahoo_auth'])
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
            except Exception as e:
                st.error(f"Secrets hatası: {e}")
                return None
        else:
            st.error("❌ oauth2.json yok!")
            return None
    # -----------------------------

    status_container.info("🚀 Yahoo sunucularına bağlanılıyor...")
    
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
            st.error(f"❌ Lig ID ({TARGET_LEAGUE_ID}) bulunamadı!")
            return None

        lg = gm.to_league(target_league_key)
        
        all_data = []
        teams = lg.teams()
        total_steps = len(teams) + 1 
        progress_bar = st.progress(0, text="Analiz başlıyor...")
        step_count = 0

        # 1. ADIM: TAKIMLARI TARA
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                
                if p_ids:
                    # 'season' = Toplam İstatistikler (GP burada kesin vardır)
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    
                    for player_meta, player_stat in zip(roster, stats):
                        process_player_totals(player_meta, player_stat, t_name, "Sahipli", all_data)
            except:
                pass
            
            step_count += 1
            progress_bar.progress(step_count / total_steps, text=f"{t_name} taranıyor...")

        # 2. ADIM: FREE AGENTLARI TARA
        try:
            progress_bar.progress(0.95, text="🆓 Free Agent havuzu taranıyor...")
            fa_players = lg.free_agents(None)[:60] 
            fa_ids = [p['player_id'] for p in fa_players]
            
            if fa_ids:
                fa_stats = lg.player_stats(fa_ids, ANALYSIS_TYPE)
                for player_meta, player_stat in zip(fa_players, fa_stats):
                    process_player_totals(player_meta, player_stat, "🆓 FREE AGENT", "Free Agent", all_data)
        except Exception as e:
            st.warning(f"FA Hatası: {e}")

        progress_bar.empty()
        status_container.empty()
        
        if not all_data:
            st.error("❌ Veri listesi boş!")
            return None
            
        return pd.DataFrame(all_data)
        
    except Exception as e:
        st.error(f"❌ GENEL HATA: {e}")
        return None

def process_player_totals(meta, stat, team_name, ownership_status, data_list):
    """
    Toplam verileri alır, GP'ye bölerek Ortalamaya çevirir.
    """
    try:
        def get_val(val):
            if val == '-' or val is None: return 0.0
            return float(val)

        # 1. GP (Maç Sayısı) - En Kritik Veri
        gp = get_val(stat.get('GP'))
        
        # Eğer GP 0 ise hesaplama yapamayız (Sıfıra bölünme hatası olur)
        if gp == 0:
            return

        # 2. MPG (Dakika) Hesaplama
        # Yahoo'da Toplam dakika bazen 'MIN' bazen 'MPG' etiketiyle gelir
        # Örnek veri: "540:30" (540 dakika 30 saniye)
        total_minutes = 0.0
        raw_min = stat.get('MIN', stat.get('MPG', '0'))
        
        if raw_min and raw_min != '-':
            raw_min = str(raw_min)
            if ":" in raw_min:
                parts = raw_min.split(":")
                total_minutes = float(parts[0]) + (float(parts[1]) / 60.0)
            else:
                total_minutes = float(raw_min)
        
        mpg = total_minutes / gp # Toplam Dakika / Maç Sayısı

        # 3. İstatistikleri Ortalamaya Çevirme
        # Yüzdeler (FG%, FT%) zaten yüzde gelir, bölünmez.
        # Sayısal değerler (PTS, REB, AST) bölünür.
        
        fg_pct = get_val(stat.get('FG%'))
        ft_pct = get_val(stat.get('FT%'))
        
        # Sakatlık Durumu
        status_code = meta.get('status', '')
        if status_code:
            status_display = f"⚠️ {status_code.upper()}"
        else:
            status_display = "✅ Sağlam"

        row = {
            'Player': meta['name'],
            'Team': team_name,
            'Owner_Status': ownership_status,
            'Injury': status_display,
            'GP': int(gp),
            'MPG': round(mpg, 1),
            'FG%': fg_pct, # Yüzdeler bölünmez
            'FT%': ft_pct, # Yüzdeler bölünmez
            '3PTM': round(get_val(stat.get('3PTM')) / gp, 1),
            'PTS': round(get_val(stat.get('PTS')) / gp, 1),
            'REB': round(get_val(stat.get('REB')) / gp, 1),
            'AST': round(get_val(stat.get('AST')) / gp, 1),
            'ST': round(get_val(stat.get('ST')) / gp, 1),
            'BLK': round(get_val(stat.get('BLK')) / gp, 1),
            'TO': round(get_val(stat.get('TO')) / gp, 1)
        }
        data_list.append(row)
    except Exception:
        pass

def calculate_z_scores(df):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    if df.empty: return df

    for cat in cats:
        if cat not in df.columns: df[cat] = 0.0
        mean = df[cat].mean()
        std = df[cat].std()
        if std == 0: std = 1
        
        col_name = f'z_{cat}'
        if cat == 'TO':
            df[col_name] = (mean - df[cat]) / std
        else:
            df[col_name] = (df[cat] - mean) / std
    return df

def analyze_team_needs(df, my_team_name):
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    z_cols = [f'z_{c}' for c in cats]
    
    my_team_df = df[df['Team'] == my_team_name]
    if my_team_df.empty: return [], []

    team_profile = my_team_df[z_cols].sum().sort_values()
    weaknesses = [w.replace('z_', '') for w in team_profile.head(4).index]
    strengths = [s.replace('z_', '') for s in team_profile.tail(3).index]
    return weaknesses, strengths

def score_players(df, targets):
    df['Skor'] = 0
    cats = ['FG%', 'FT%', '3PTM', 'PTS', 'REB', 'AST', 'ST', 'BLK', 'TO']
    for cat in cats:
        if f'z_{cat}' in df.columns:
            weight = 3.0 if cat in targets else 1.0
            df['Skor'] += df[f'z_{cat}'] * weight
    return df

# --- ARAYÜZ ---

st.title("🏀 Burak's Wizards - GM Paneli")
st.markdown("**Veri Kaynağı:** Sezon Toplamlarından Hesaplanan Ortalamalar (Kesin Veri)")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_team_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"📉 **Eksiklerin:** {', '.join(targets)}")
        with col2:
            st.success(f"📈 **Güçlerin:** {', '.join(strengths)}")

        st.markdown("---")
        
        # FİLTRELEME
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
             filter_status = st.multiselect("Oyuncu Tipi:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"])
        with col_filter2:
             hide_injured = st.checkbox("Sakatları Gizle (INJ/O)", value=False)

        # Filtre Uygulama
        filtered_df = df.copy()
        if filter_status:
            filtered_df = filtered_df[filtered_df['Owner_Status'].isin(filter_status)]
        
        if hide_injured:
            filtered_df = filtered_df[filtered_df['Injury'].str.contains("Sağlam|DTD")]

        tab1, tab2, tab3 = st.tabs(["🔥 Hedef Oyuncular", "📋 Kadrom", "🌍 Tüm Liste"])

        with tab1:
            st.subheader("En İyi Adaylar")
            trade_df = filtered_df[filtered_df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            
            st.dataframe(
                trade_df[['Player', 'Team', 'Injury', 'GP', 'MPG', 'Skor'] + targets].head(30),
                column_config={
                    "Skor": st.column_config.ProgressColumn("Uygunluk", format="%.1f", min_value=0, max_value=trade_df['Skor'].max()),
                    "Injury": st.column_config.TextColumn("Sağlık"),
                    "MPG": st.column_config.NumberColumn("Dakika (Ort)", format="%.1f"),
                    "GP": st.column_config.NumberColumn("Maç"),
                },
                use_container_width=True
            )
            
        with tab2:
            my_team_df = df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(
                my_team_df[['Player', 'Injury', 'GP', 'MPG', 'Skor', 'PTS', 'REB', 'AST', 'ST', 'BLK']], 
                use_container_width=True
            )

        with tab3:
            st.dataframe(filtered_df)
else:
    st.info("⚠️ Veri bekleniyor...")
