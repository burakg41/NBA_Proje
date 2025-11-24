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
import concurrent.futures  # PARALEL İŞLEM İÇİN KRİTİK KÜTÜPHANE

# --- NBA API ---
from nba_api.stats.endpoints import leaguedashplayerstats, scoreboardv2

# --- AYARLAR ---
SEASON_YEAR = 2025
NBA_SEASON_STRING = "2025-26"
TARGET_LEAGUE_ID = "61142"
MY_TEAM_NAME = "Burak's Wizards"
ANALYSIS_TYPE = "average_season"
CACHE_FILE = "optimized_data_cache.json"  # Verileri buraya yedekleyeceğiz
CACHE_DURATION_HOURS = 4  # 4 Saatte bir yenile

st.set_page_config(page_title="Burak's GM Dashboard", layout="wide")

# --- YAN PANEL ---
with st.sidebar:
    st.header("⚡ GM Kontrol Paneli")
    if st.button("🚀 Zorla Yenile (API)"):
        st.cache_data.clear()
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.rerun()

    st.markdown("---")
    st.subheader("🎯 Strateji")
    punt_cats = st.multiselect(
        "Punt:",
        ["FG%", "FT%", "3PTM", "PTS", "REB", "AST", "ST", "BLK", "TO"],
    )
    st.caption(
        f"Veri Ömrü: {CACHE_DURATION_HOURS} Saat. Önbellekten okununca hızlanır."
    )

# ==========================================
# 1. YAHOO AUTH HELPER (ÖNEMLİ FIX)
# ==========================================


def build_yahoo_oauth():
    """
    oauth2.json + st.secrets['yahoo_auth'] kombinasyonunu daha toleranslı yönetir.

    Desteklenen senaryolar:
    1) oauth2.json zaten var ve içinde access_token dahil her şey var -> direkt kullan
    2) secrets içinde tam token seti var (access_token, refresh_token, token_time vs.) -> dosyaya yaz ve kullan
    3) secrets içinde sadece consumer_key / consumer_secret var -> quickstart senaryosu (ilk kez login)
    """

    # 1) Varolan dosyayı kullanmayı dene
    if os.path.exists("oauth2.json"):
        try:
            return OAuth2(None, None, from_file="oauth2.json")
        except KeyError:
            # Dosya bozuk / eksik -> silip sıfırdan dene
            try:
                os.remove("oauth2.json")
            except OSError:
                pass  # yoksa problem değil

    # 2) Secrets yoksa net uyarı ver
    if "yahoo_auth" not in st.secrets:
        st.error(
            "Yahoo OAuth bilgileri bulunamadı.\n\n"
            "Streamlit secrets'e `[yahoo_auth]` bölümünde en az "
            "`consumer_key` ve `consumer_secret` eklemen gerekiyor."
        )
        return None

    s = dict(st.secrets["yahoo_auth"])

    # token_time string geldiyse floata çevirelim (varsayımsal)
    if "token_time" in s:
        try:
            s["token_time"] = float(s["token_time"])
        except Exception:
            pass

    # 2a) Secrets içinde zaten access_token varsa → tam oauth2.json gibi davran
    if "access_token" in s:
        try:
            with open("oauth2.json", "w") as f:
                json.dump(s, f)
            return OAuth2(None, None, from_file="oauth2.json")
        except Exception as e:
            st.error(f"Yahoo OAuth dosyası yazılırken hata oluştu: {e}")
            return None

    # 2b) Sadece consumer_key / consumer_secret varsa → Quickstart senaryosu
    if "consumer_key" in s and "consumer_secret" in s:
        try:
            # Quickstart: önce sadece ck/cs içeren bir json dosyası oluştur
            creds = {
                "consumer_key": s["consumer_key"],
                "consumer_secret": s["consumer_secret"],
            }
            with open("oauth2.json", "w") as f:
                json.dump(creds, f)

            # Bu adımda yahoo_oauth ilk kez çalışırken tarayıcıda yetki isteyebilir
            oauth = OAuth2(None, None, from_file="oauth2.json")
            return oauth

        except KeyError as e:
            st.error(
                f"Yahoo OAuth yapılandırmasında eksik anahtar var: {e}\n"
                "Lütfen `consumer_key` ve `consumer_secret` değerlerini kontrol et."
            )
            return None
        except Exception as e:
            st.error(f"Yahoo OAuth başlatılırken hata oluştu: {e}")
            return None

    # 2c) Buraya düştüyse secrets formatı yanlış
    st.error(
        "st.secrets['yahoo_auth'] formatı hatalı.\n\n"
        "Geçerli örnek:\n\n"
        "[yahoo_auth]\n"
        "consumer_key = \"...\"\n"
        "consumer_secret = \"...\"\n"
        "# (İstersen access_token, refresh_token, token_time vs. de ekleyebilirsin)"
    )
    return None


# ==========================================
# 2. OPTİMİZE EDİLMİŞ VERİ ÇEKME MOTORU
# ==========================================


def fetch_nba_schedule():
    """NBA Fikstürünü çeker (Hafif işlem)"""
    try:
        team_game_counts = {}
        today = datetime.now()
        # Sadece önümüzdeki 7 güne bakıyoruz, paralel yapmaya gerek yok, hızlıdır.
        for i in range(7):
            date_str = (today + timedelta(days=i)).strftime("%m/%d/%Y")
            board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=2)  # Timeout ekledik
            line_score = board.line_score.get_data_frame()
            if not line_score.empty:
                playing_teams = line_score["TEAM_ABBREVIATION"].unique()
                for team in playing_teams:
                    team_game_counts[team] = team_game_counts.get(team, 0) + 1
        return team_game_counts
    except Exception:
        return {}


def fetch_nba_stats():
    """NBA.com İstatistikleri"""
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=NBA_SEASON_STRING, per_mode_detailed="PerGame", timeout=5
        )
        df = stats.get_data_frames()[0]
        nba_data = {}
        for _, row in df.iterrows():
            clean_name = row["PLAYER_NAME"].lower().replace(".", "").strip()
            nba_data[clean_name] = {
                "GP": row["GP"],
                "MPG": row["MIN"],
                "TEAM": row["TEAM_ABBREVIATION"],
            }
        return nba_data
    except Exception:
        return {}


# --- PARALEL İŞLEM İÇİN YARDIMCI FONKSİYON ---
def fetch_team_roster_data(team_key, team_name, lg):
    """Tek bir takımın verisini çeker (Thread içinde çalışacak)"""
    try:
        roster = lg.to_team(team_key).roster()
        p_ids = [p["player_id"] for p in roster]
        if not p_ids:
            return []

        # Çift Sorgu (Sezon + Son 1 Ay)
        stats_s = lg.player_stats(p_ids, ANALYSIS_TYPE)
        stats_m = lg.player_stats(p_ids, "last_month")

        team_data = []
        for i, pm in enumerate(roster):
            # Veriyi işle ve dict olarak döndür (DataFrame değil, daha hızlı)
            p_data = process_player_raw(pm, stats_s[i], stats_m[i], team_name, "Sahipli")
            if p_data:
                team_data.append(p_data)
        return team_data
    except Exception:
        return []


def fetch_fa_chunk(chunk_ids, chunk_players, lg):
    """Free Agent parçasını çeker"""
    try:
        stats_s = lg.player_stats(chunk_ids, ANALYSIS_TYPE)
        stats_m = lg.player_stats(chunk_ids, "last_month")
        chunk_data = []
        for k, pm in enumerate(chunk_players):
            p_data = process_player_raw(pm, stats_s[k], stats_m[k], "🆓 FA", "Free Agent")
            if p_data:
                chunk_data.append(p_data)
        return chunk_data
    except Exception:
        return []


def process_player_raw(meta, stat_s, stat_m, team_name, ownership):
    """Veriyi ham haliyle işler (NBA verisiyle sonra birleşecek)"""
    try:
        def get_val(val):
            return float(val) if val not in ["-", None] else 0.0

        p_name = meta["name"]
        st_code = meta.get("status", "")
        raw_pos = meta.get("display_position", "")

        pts_s, pts_m = get_val(stat_s.get("PTS")), get_val(stat_m.get("PTS"))
        trend = pts_m - pts_s if pts_m > 0 else 0

        return {
            "Player": p_name,
            "Team": team_name,
            "Yahoo_Team": meta.get("editorial_team_abbr", "N/A").upper(),
            "Owner_Status": ownership,
            "Raw_Pos": raw_pos,
            "Status": st_code,
            "Trend_Score": trend,
            "FG%": get_val(stat_s.get("FG%")) * 100,
            "FT%": get_val(stat_s.get("FT%")) * 100,
            "3PTM": get_val(stat_s.get("3PTM")),
            "PTS": pts_s,
            "REB": get_val(stat_s.get("REB")),
            "AST": get_val(stat_s.get("AST")),
            "ST": get_val(stat_s.get("ST")),
            "BLK": get_val(stat_s.get("BLK")),
            "TO": get_val(stat_s.get("TO")),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def master_data_loader():
    """TÜM SİSTEMİN BEYNİ: Önce diske bakar, yoksa paralel API çağrısı yapar"""

    # 1. DİSK KONTROLÜ (HIZLI BAŞLANGIÇ)
    if os.path.exists(CACHE_FILE):
        file_age = time.time() - os.path.getmtime(CACHE_FILE)
        if file_age < (CACHE_DURATION_HOURS * 3600):
            st.toast("⚡ Veriler diskten yüklendi (Turbo Mod)", icon="🚀")
            try:
                with open(CACHE_FILE, "r") as f:
                    return pd.DataFrame(json.load(f)), None  # Lig objesi diskten dönmez, gerekirse tekrar bağlanırız
            except Exception:
                pass  # Hata varsa API'ye geç

    # 2. API BAĞLANTISI (YAHOO OAUTH)
    sc = build_yahoo_oauth()
    if sc is None:
        # build_yahoo_oauth detaylı hata mesajını zaten bastı
        return None, None

    try:
        if not sc.token_is_valid():
            sc.refresh_access_token()
    except Exception as e:
        st.error(f"Yahoo token yenilenirken hata oluştu: {e}")
        return None, None

    try:
        gm = yfa.Game(sc, "nba")
    except Exception as e:
        st.error(f"Yahoo Fantasy Game nesnesi oluşturulamadı: {e}")
        return None, None

    try:
        lid = next((l for l in gm.league_ids(year=SEASON_YEAR) if TARGET_LEAGUE_ID in l), None)
        if not lid:
            st.error("Hedef Yahoo lig ID'si bulunamadı. TARGET_LEAGUE_ID değerini kontrol et.")
            return None, None
        lg = gm.to_league(lid)

        # 3. PARALEL VERİ ÇEKME (THREADING)
        status_text = st.empty()
        status_text.info("🌐 API'lere Paralel Bağlanılıyor (NBA + Yahoo)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # A. NBA Verilerini Başlat
            future_nba = executor.submit(fetch_nba_stats)
            future_sched = executor.submit(fetch_nba_schedule)

            # B. Yahoo Takımlarını Başlat
            teams = lg.teams()
            future_teams = {
                executor.submit(fetch_team_roster_data, k, v["name"], lg): v["name"]
                for k, v in teams.items()
            }

            # C. Yahoo Free Agents
            fa_players = lg.free_agents(None)[:300]
            fa_ids = [p["player_id"] for p in fa_players]

            all_raw_data = []

            # Takım Verilerini Topla
            for future in concurrent.futures.as_completed(future_teams):
                data = future.result()
                if data:
                    all_raw_data.extend(data)

            # FA Verilerini Çek (Chunked)
            chunk_size = 25
            for i in range(0, len(fa_ids), chunk_size):
                chunk_ids = fa_ids[i : i + chunk_size]
                chunk_players = fa_players[i : i + chunk_size]
                chunk_data = fetch_fa_chunk(chunk_ids, chunk_players, lg)
                if chunk_data:
                    all_raw_data.extend(chunk_data)

            # NBA Sonuçlarını Bekle
            nba_stats = future_nba.result()
            nba_sched = future_sched.result()

        # 4. VERİ BİRLEŞTİRME (MERGE)
        status_text.text("Veriler birleştiriliyor...")

        final_data = []
        for p in all_raw_data:
            # İsim Temizliği
            clean_name = p["Player"].lower().replace(".", "").strip()

            # NBA Verisi Ekle
            real_gp = nba_stats.get(clean_name, {}).get("GP", 0)
            real_mpg = nba_stats.get(clean_name, {}).get("MPG", 0.0)

            # Takım Eşleşmesi (Yahoo veya NBA'den)
            real_team = nba_stats.get(clean_name, {}).get("TEAM", p["Yahoo_Team"])
            if real_team == "PHO":
                real_team = "PHX"

            # Fikstür
            games_next = nba_sched.get(real_team, 0)

            # Görselleştirmeler
            simple_pos = (
                p["Raw_Pos"]
                .replace("PG", "G")
                .replace("SG", "G")
                .replace("SF", "F")
                .replace("PF", "F")
            )
            u_pos = sorted(
                list(set(simple_pos.split(","))),
                key=lambda x: 1 if x == "G" else (2 if x == "F" else 3),
            )

            # Trend İkon
            if p["Trend_Score"] > 3.0:
                ti = "🔥"
            elif p["Trend_Score"] > 1.0:
                ti = "↗️"
            elif p["Trend_Score"] < -3.0:
                ti = "🥶"
            elif p["Trend_Score"] < -1.0:
                ti = "↘️"
            else:
                ti = "➖"

            # Sağlık İkon
            if p["Status"] in ["INJ", "O"]:
                hi = f"🟥 {p['Status']}"
            elif p["Status"] in ["GTD", "DTD"]:
                hi = f"Rx {p['Status']}"
            else:
                hi = "✅"

            p["Pos"] = ",".join(u_pos)
            p["Health"] = hi
            p["Trend"] = ti
            p["GP"] = int(real_gp)
            p["MPG"] = float(real_mpg)
            p["Real_Team"] = real_team
            p["Games_Next_7D"] = int(games_next)

            # Gereksizleri at
            del p["Raw_Pos"], p["Status"], p["Trend_Score"], p["Yahoo_Team"]
            final_data.append(p)

        df = pd.DataFrame(final_data)

        # 5. DİSKE YEDEKLE
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(final_data, f)
        except Exception:
            pass

        status_text.empty()
        return df, lg

    except Exception as e:
        st.error(f"Kritik Hata: {e}")
        return None, None


# ==========================================
# 3. HESAPLAMA MOTORU (CACHED OLMALI)
# ==========================================


def perform_calculations(df, punt_list):
    # Z-Score
    cats = ["FG%", "FT%", "3PTM", "PTS", "REB", "AST", "ST", "BLK", "TO"]
    active_cats = [c for c in cats if c not in punt_list]

    for c in cats:
        if c in punt_list:
            df[f"z_{c}"] = 0.0
            continue
        if c not in df.columns:
            df[c] = 0.0
        mean, std = df[c].mean(), df[c].std()
        if std == 0:
            std = 1
        df[f"z_{c}"] = (mean - df[c]) / std if c == "TO" else (df[c] - mean) / std

    df["Genel_Kalite"] = df[[f"z_{c}" for c in active_cats]].sum(axis=1)
    return df, active_cats


def analyze_needs(df, my_team, active_cats):
    z_cols = [f"z_{c}" for c in active_cats]
    m_df = df[df["Team"] == my_team]
    if m_df.empty:
        return [], []
    prof = m_df[z_cols].sum().sort_values()
    return [x.replace("z_", "") for x in prof.head(4).index], [
        x.replace("z_", "") for x in prof.tail(3).index
    ]


def score_players(df, targets, active_cats):
    df["Skor"] = 0
    for c in active_cats:
        if f"z_{c}" in df.columns:
            w = 3.0 if c in targets else 1.0
            df["Skor"] += df[f"z_{c}"] * w
    # Fikstür Etkisi (Streaming için)
    df["Skor"] += df["Games_Next_7D"].apply(
        lambda x: 1.0 if x >= 4 else (-0.5 if x <= 2 else 0)
    )
    return df


# --- RAKİP ANALİZİ (BASİT) ---
def get_matchup_prediction(df, my_team, opp_team, cats):
    my_stats = df[df["Team"] == my_team][cats].mean()
    opp_stats = df[df["Team"] == opp_team][cats].mean()
    s_me, s_opp, details = 0, 0, []
    for c in cats:
        val_me, val_opp = my_stats[c], opp_stats[c]
        win = (val_me < val_opp) if c == "TO" else (val_me > val_opp)
        if win:
            s_me += 1
        else:
            s_opp += 1
        details.append(
            {
                "Kat": c,
                "Ben": f"{val_me:.1f}",
                "Rakip": f"{val_opp:.1f}",
                "Sonuç": "✅" if win else "❌",
            }
        )
    return s_me, s_opp, pd.DataFrame(details)


# --- TAKAS MOTORU (OPTIMIZE EDİLMİŞ) ---
def trade_engine_optimized(df, my_team):
    my_assets = df[df["Team"] == my_team].sort_values("Skor").head(5)
    # Sadece ilk 10 rakip oyuncuya bak (Hız için)
    opp_assets = (
        df[(df["Team"] != my_team) & (df["Owner_Status"] == "Sahipli")]
        .sort_values("Skor", ascending=False)
        .head(10)
    )

    proposals = []
    # 1v1
    for _, m in my_assets.iterrows():
        for _, o in opp_assets.iterrows():
            diff = m["Genel_Kalite"] - o["Genel_Kalite"]
            gain = o["Skor"] - m["Skor"]
            if diff > -3.0 and gain > 2.0:
                proposals.append(
                    {
                        "Tür": "1v1",
                        "Ver": m["Player"],
                        "Al": o["Player"],
                        "Takım": o["Team"],
                        "Şans": diff,
                        "Kazanç": gain,
                    }
                )

    # 2v1 (Konsolidasyon) - Sadece en iyi 2 asset kombinasyonu
    if len(my_assets) >= 2:
        m_combos = list(itertools.combinations(my_assets.head(3).index, 2))
        for idxs in m_combos:
            m_list = [df.loc[i] for i in idxs]
            tot_give_q = sum([p["Genel_Kalite"] for p in m_list])
            tot_give_s = sum([p["Skor"] for p in m_list])

            for _, o in opp_assets.head(5).iterrows():
                diff = tot_give_q - o["Genel_Kalite"]
                gain = o["Skor"] - tot_give_s
                # Süperstar almak için değerinden fazla vermeye (overpay) razıyız (diff > -5)
                # Ama aldığımız oyuncunun skoru bize çok şey katmalı (Çünkü 1 slot açıyoruz)
                # Not: Slot açmak +Skor demektir (FA'dan adam alırsın). O yüzden gain düşük bile olsa kardır.
                if diff > -4.0 and gain > -5.0:
                    proposals.append(
                        {
                            "Tür": "2v1",
                            "Ver": f"{m_list[0]['Player']}, {m_list[1]['Player']}",
                            "Al": o["Player"],
                            "Takım": o["Team"],
                            "Şans": diff,
                            "Kazanç": gain + 5.0,  # +5 Slot Bonusu
                        }
                    )

    if not proposals:
        return pd.DataFrame()

    return pd.DataFrame(proposals).sort_values("Kazanç", ascending=False)


# ==========================================
# 4. DASHBOARD (MAIN)
# ==========================================

st.title("🚀 Burak's GM Dashboard (Turbo Mod)")
st.markdown("---")

df_raw, lg = master_data_loader()

if df_raw is not None and not df_raw.empty:
    df_calc, active_cats = perform_calculations(df_raw.copy(), punt_cats)
    targets, strengths = analyze_needs(df_calc, MY_TEAM_NAME, active_cats)

    if targets:
        df_final = score_players(df_calc, targets, active_cats)

        # Filtreleme
        col1, col2 = st.columns(2)
        f_stat = col1.multiselect(
            "Filtre:", ["Sahipli", "Free Agent"], default=["Sahipli", "Free Agent"]
        )
        h_inj = col2.checkbox("Sakatları Gizle (✅)", value=True)

        v_df = df_final.copy()
        if f_stat:
            v_df = v_df[v_df["Owner_Status"].isin(f_stat)]
        if h_inj:
            v_df = v_df[v_df["Health"].str.contains("✅")]

        show_cols = [
            "Player",
            "Team",
            "Real_Team",
            "Games_Next_7D",
            "Trend",
            "Pos",
            "Health",
            "GP",
            "MPG",
            "Skor",
            "FG%",
            "FT%",
            "3PTM",
            "PTS",
            "REB",
            "AST",
            "ST",
            "BLK",
            "TO",
        ]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🔥 Hedefler", "📋 Kadrom", "🌍 Liste", "🔄 Takas", "⚔️ Rakip"]
        )

        with tab1:
            st.caption("Fikstür Avantajı: Sarı = Çok Maç, Kırmızı = Az Maç")
            trade_df = v_df[v_df["Team"] != MY_TEAM_NAME].sort_values(
                by="Skor", ascending=False
            )
            st.dataframe(
                trade_df[show_cols].head(50),
                column_config={
                    "Skor": st.column_config.ProgressColumn(
                        "Puan",
                        format="%.1f",
                        max_value=float(trade_df["Skor"].max())
                        if not trade_df["Skor"].empty
                        else 0,
                    ),
                    "Games_Next_7D": st.column_config.NumberColumn(
                        "7G Maç", format="%d 🏀"
                    ),
                },
                use_container_width=True,
            )

            # Scatter Plot (Performans vs Fikstür)
            fa_view = v_df[v_df["Owner_Status"] == "Free Agent"].copy()
            if not fa_view.empty:
                fig = px.scatter(
                    fa_view.head(100),
                    x="MPG",
                    y="Skor",
                    color="Games_Next_7D",
                    size="GP",
                    hover_name="Player",
                    title="Free Agent Fırsat Haritası",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(
                df_final[df_final["Team"] == MY_TEAM_NAME]
                .sort_values("Skor", ascending=False)[show_cols],
                use_container_width=True,
            )

        with tab3:
            st.dataframe(v_df[show_cols], use_container_width=True)

        with tab4:
            if st.button("Takasları Tara"):
                res = trade_engine_optimized(df_final, MY_TEAM_NAME)
                if res is not None and not res.empty:
                    st.dataframe(
                        res,
                        column_config={
                            "Şans": st.column_config.ProgressColumn(
                                "Onay İhtimali", min_value=-5, max_value=5
                            )
                        },
                        use_container_width=True,
                    )
                else:
                    st.warning("Fırsat bulunamadı.")

        with tab5:
            # Rakip Analizi
            opps = df_final[df_final["Owner_Status"] == "Sahipli"]["Team"].unique()
            opps = [o for o in opps if o != MY_TEAM_NAME]
            if opps:
                sel = st.selectbox("Rakip:", opps)
                if sel:
                    s1, s2, det = get_matchup_prediction(
                        df_final, MY_TEAM_NAME, sel, active_cats
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Sen", s1)
                    c2.metric(sel, s2)
                    st.dataframe(det, use_container_width=True)
            else:
                st.info("Rakip takımlar yüklenemedi.")

else:
    st.info("Sistem başlatılamadı. OAuth / lig bilgilerini kontrol et lütfen.")
