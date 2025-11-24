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
ANALYSIS_TYPE = 'average_season' 

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- ÖNBELLEĞİ TEMİZLEME BUTONU ---
with st.sidebar:
    st.header("Yönetim Paneli")
    if st.button("🔄 Verileri Yenile & Önbelleği Sil"):
        st.cache_data.clear()
        st.rerun()
    st.info(f"📅 Analiz Modu: Sezon Ortalamaları")

# --- VERİ YÜKLEME FONKSİYONU ---
@st.cache_data(ttl=3600)
def load_data():
    status_container = st.empty()
    
    # ==========================================
    # --- DEBUG / HATA AYIKLAMA BAŞLANGIÇ ---
    # ==========================================
    st.markdown("### 🛠️ DEBUG PENCERESİ")
    try:
        # Mevcut anahtarları göster (Değerleri gösterme, güvenlik için)
        available_keys = list(st.secrets.keys())
        st.write(f"Mevcut Secret Anahtarları: {available_keys}")
        
        if 'yahoo_auth' in st.secrets:
            st.success("✅ [yahoo_auth] anahtarı algılandı!")
            # İçindeki zorunlu alanları kontrol et
            auth_keys = st.secrets['yahoo_auth']
            required = ['consumer_key', 'consumer_secret', 'access_token']
            missing = [k for k in required if k not in auth_keys]
            if missing:
                st.error(f"❌ Eksik Bilgiler Var: {missing}")
            else:
                st.info("✅ Gerekli tüm alt anahtarlar mevcut.")
        else:
            st.error("❌ [yahoo_auth] anahtarı BULUNAMADI. Secrets ayarlarını kontrol et.")
    except Exception as e:
        st.error(f"Debug sırasında hata: {e}")
    st.markdown("---")
    # ==========================================
    # --- DEBUG BİTİŞ ---
    # ==========================================

    # --- BULUT İÇİN GİZLİ DOSYA YARATMA ---
    if not os.path.exists('oauth2.json'):
        if 'yahoo_auth' in st.secrets:
            try:
                # Secrets verisini JSON formatına çevirip dosyaya yazıyoruz
                secrets_dict = dict(st.secrets['yahoo_auth'])
                
                # token_time sayı olmalı, kontrol edelim
                if 'token_time' in secrets_dict:
                     secrets_dict['token_time'] = float(secrets_dict['token_time'])
                
                with open('oauth2.json', 'w') as f:
                    json.dump(secrets_dict, f)
                st.caption("🔑 oauth2.json dosyası başarıyla oluşturuldu.")
            except Exception as e:
                st.error(f"Secrets dosya oluşturma hatası: {e}")
                return None
        else:
            st.error("❌ HATA: 'oauth2.json' bulunamadı ve Secrets ayarlanmamış!")
            return None
    # -------------------------------------------------------------

    status_container.info("🚀 Yahoo sunucularına bağlanılıyor...")
    
    try:
        # 1. Bağlantı
        sc = OAuth2(None, None, from_file='oauth2.json')
        if not sc.token_is_valid():
            sc.refresh_access_token()
        gm = yfa.Game(sc, 'nba')
        
        # 2. Lig Bulma
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
        
        # 3. Verileri Çekme
        teams = lg.teams()
        all_data = []
        
        progress_text = "Veriler analiz ediliyor..."
        my_bar = st.progress(0, text=progress_text)
        total_teams = len(teams)
        count = 0
        
        for team_key in teams.keys():
            t_name = teams[team_key]['name']
            
            try:
                roster = lg.to_team(team_key).roster()
                p_ids = [p['player_id'] for p in roster]
                
                if p_ids:
                    stats = lg.player_stats(p_ids, ANALYSIS_TYPE)
                    
                    for p_stat in stats:
                        try:
                            def get_val(val):
                                if val == '-' or val is None: return 0.0
                                return float(val)

                            gp = get_val(p_stat.get('GP'))
                            pts = get_val(p_stat.get('PTS'))
                            
                            # Hiç oynamamış oyuncuyu ele
                            if gp == 0 and pts == 0:
                                continue

                            row = {
                                'Player': p_stat['name'],
                                'Team': t_name,
                                'GP': gp,
                                'FG%': get_val(p_stat.get('FG%')),
                                'FT%': get_val(p_stat.get('FT%')),
                                '3PTM': get_val(p_stat.get('3PTM')),
                                'PTS': pts,
                                'REB': get_val(p_stat.get('REB')),
                                'AST': get_val(p_stat.get('AST')),
                                'ST': get_val(p_stat.get('ST')),
                                'BLK': get_val(p_stat.get('BLK')),
                                'TO': get_val(p_stat.get('TO'))
                            }
                            all_data.append(row)
                        except:
                            continue

            except Exception:
                pass
            
            count += 1
            my_bar.progress(count / total_teams, text=f"{t_name} tamamlandı...")
            
        my_bar.empty()
        status_container.empty()
        
        if not all_data:
            st.error("❌ Veri listesi boş! API yanıt vermedi.")
            return None
            
        return pd.DataFrame(all_data)
        
    except Exception as e:
        st.error(f"❌ GENEL HATA: {e}")
        return None

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
    if my_team_df.empty: 
        st.warning(f"⚠️ '{my_team_name}' takımı verilerde bulunamadı.")
        return [], []

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
st.markdown(f"**Veri Kaynağı:** 2025-2026 Sezon Ortalamaları (21 Ekim - Bugün)")
st.markdown("---")

df = load_data()

if df is not None and not df.empty:
    df = calculate_z_scores(df)
    targets, strengths = analyze_team_needs(df, MY_TEAM_NAME)
    
    if targets:
        df = score_players(df, targets)
        
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"📉 **Takımının Eksikleri:** {', '.join(targets)}")
        with col2:
            st.success(f"📈 **Takımının Güçleri:** {', '.join(strengths)}")

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["🔥 Takas Önerileri", "📋 Benim Kadrom", "🌍 Tüm Lig"])

        with tab1:
            st.subheader("Hedef Oyuncular (Takas)")
            st.caption("Eksiklerini kapatacak en iyi oyuncular:")
            
            trade_df = df[df['Team'] != MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            
            st.dataframe(
                trade_df[['Player', 'Team', 'Skor'] + targets].head(20),
                column_config={
                    "Skor": st.column_config.ProgressColumn(
                        "Uygunluk", format="%.1f", min_value=0, max_value=trade_df['Skor'].max()
                    ),
                },
                use_container_width=True
            )
            
        with tab2:
            st.subheader("Takım Analizin")
            my_team_df = df[df['Team'] == MY_TEAM_NAME].sort_values(by='Skor', ascending=False)
            st.dataframe(my_team_df, use_container_width=True)

        with tab3:
            st.dataframe(df)
            
else:
    st.info("⚠️ Veri bekleniyor...")
