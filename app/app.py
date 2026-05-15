"""P2 — Indonesia Mobility Flow Explorer v3 — Elegant Minimal"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
import os, json

st.set_page_config(
    page_title="Indonesia Mobility Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.hero {
    padding: 3rem 0 2.5rem;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: .65rem; font-weight: 500; letter-spacing: .14em;
    text-transform: uppercase; color: #4b5563; margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.75rem; font-weight: 600; line-height: 1.15;
    color: #f1f5f9; letter-spacing: -.01em; margin: 0 0 .75rem;
}
.hero-title em { font-style: italic; color: #60a5fa; }
.hero-sub {
    font-size: .8rem; color: #4b5563; letter-spacing: .02em;
}
.kpi {
    background: #0d1117; border: 1px solid #1e2533;
    border-radius: 6px; padding: 1.25rem 1.5rem;
}
.kpi-label {
    font-size: .62rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: .1em; color: #374151; margin-bottom: .6rem;
}
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 600; color: #f9fafb; line-height: 1;
}
.kpi-value.neg { color: #f87171; }
.kpi-sub { font-size: .68rem; color: #1f2937; margin-top: .3rem; }
.kpi-rule { height: 1px; margin-top: .875rem; opacity: .6; }
.kpi-rule.blue  { background: linear-gradient(90deg,#3b82f6,transparent); }
.kpi-rule.red   { background: linear-gradient(90deg,#ef4444,transparent); }
.kpi-rule.amber { background: linear-gradient(90deg,#f59e0b,transparent); }
.kpi-rule.green { background: linear-gradient(90deg,#10b981,transparent); }
.sec {
    font-size: .62rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .1em; color: #374151;
    border-bottom: 1px solid #1e2533;
    padding-bottom: .4rem; margin: 0 0 1rem;
}
.insight {
    background: #080c12; border: 1px solid #1e2533;
    border-left: 2px solid #3b82f6;
    border-radius: 3px; padding: .75rem 1.125rem;
    margin: .875rem 0; font-size: .78rem;
    color: #6b7280; line-height: 1.7;
}
.insight.warn { border-left-color: #f59e0b; }
.insight.ok   { border-left-color: #10b981; }
[data-testid="stSidebar"] {
    background: #080c12 !important;
    border-right: 1px solid #111827;
}
[data-testid="stSidebar"] label {
    font-size: .62rem !important; color: #374151 !important;
    text-transform: uppercase; letter-spacing: .08em;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #1e2533; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #374151;
    font-size: .75rem; font-weight: 500;
    letter-spacing: .04em; padding: .6rem 1.25rem; border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #e5e7eb !important;
    border-bottom: 1px solid #e5e7eb !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── THEME ────────────────────────────────────────────────────
BG  = 'rgba(0,0,0,0)'
PBG = 'rgba(8,12,18,0.7)'
AX  = dict(gridcolor='#1e2533', linecolor='#1e2533', zerolinecolor='#1e2533')
FONT= dict(family='Inter', color='#9ca3af', size=11)

def theme(fig, h=380, margin=None, legend=None, **kw):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PBG,
        font=FONT, height=h,
        margin=margin or dict(l=0,r=0,t=20,b=10),
        legend=legend or dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#6b7280')),
        **kw
    )
    fig.update_xaxes(**AX)
    fig.update_yaxes(**AX)
    return fig

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d        = os.path.join(base, 'data', 'processed')
    sp       = pd.read_csv(os.path.join(d, 'bps_migrasi_risen_provinsi_sp2020.csv'))
    mkab     = pd.read_csv(os.path.join(d, 'meta_mobility_agregat_kabupaten.csv'))
    mbln     = pd.read_csv(os.path.join(d, 'meta_mobility_tren_bulanan.csv'))
    kom      = pd.read_csv(os.path.join(d, 'bps_komuter_provinsi_2024.csv'))
    hist     = pd.read_csv(os.path.join(d, 'bps_migrasi_historis_1980_2015.csv'))
    sus19    = pd.read_csv(os.path.join(d, 'bps_migran_risen_2019_susenas.csv'))

    magg = (mkab.groupby('polygon_name')
                .agg(mob_mean=('movement_change_mean','mean'),
                     stay_put=('stay_put_mean','mean'),
                     mob_min=('movement_change_min','mean'),
                     mob_max=('movement_change_max','mean'))
                .reset_index())

    spatial = os.path.join(os.path.dirname(base),
                           'spatial-base-system','data','spatial','processed')
    gpkg_kab = os.path.join(spatial, 'idn_kabupaten_2023_gadm.gpkg')
    if os.path.exists(gpkg_kab):
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_kab)[['nama_kab','nama_prov']]
        gdf['key'] = gdf['nama_kab'].str.strip().str.title()
        magg['key'] = magg['polygon_name'].str.strip().str.title()
        magg = magg.merge(gdf[['key','nama_prov']], on='key', how='left')
        magg = magg.rename(columns={'nama_prov':'provinsi'})
    else:
        magg['provinsi'] = None

    sp['provinsi'] = sp['provinsi'].str.strip()
    mbln['date'] = pd.to_datetime(
        mbln['tahun'].astype(str) + '-' +
        mbln['bulan'].astype(str).str.zfill(2) + '-01')
    mbln = mbln.sort_values('date')
    return sp, magg, mbln, kom, hist, sus19

sp, magg, mbln, kom, hist, sus19 = load_data()

# Province name normalization
NORM = {'DKI Jakarta':'Jakarta Raya','DI Yogyakarta':'Yogyakarta',
        'Kep. Bangka Belitung':'Bangka Belitung'}

@st.cache_data
def build_meta_prov(_magg):
    if _magg['provinsi'].notna().sum() == 0:
        return pd.DataFrame(columns=['provinsi','meta_mean'])
    return (_magg.dropna(subset=['provinsi'])
                 .groupby('provinsi')['mob_mean']
                 .mean().reset_index()
                 .rename(columns={'mob_mean':'meta_mean'}))

meta_prov = build_meta_prov(magg)

@st.cache_data
def build_join(_sp, _meta_prov):
    def n(s):
        return str(s).upper().replace('PROVINSI ','').replace('DI ','') \
                     .replace('DAERAH ISTIMEWA ','').strip()
    a = _sp.copy(); a['key'] = a['provinsi'].apply(n)
    a['key'] = a['key'].replace({'DKI JAKARTA':'JAKARTA RAYA'})
    b = _meta_prov.rename(columns={'provinsi':'prov_meta'})
    b['key'] = b['prov_meta'].apply(n)
    return a.merge(b, on='key', how='inner')

jn = build_join(sp, meta_prov)

COORDS = {
    'Aceh':(4.69,96.75),'Sumatera Utara':(2.12,99.54),
    'Sumatera Barat':(-0.74,100.21),'Riau':(0.29,101.70),
    'Kepulauan Riau':(3.94,108.14),'Jambi':(-1.61,103.61),
    'Sumatera Selatan':(-3.32,104.91),'Bengkulu':(-3.79,102.26),
    'Lampung':(-4.56,105.40),'Bangka Belitung':(-2.74,106.44),
    'Banten':(-6.40,106.12),'Jakarta Raya':(-6.21,106.84),
    'Jawa Barat':(-7.09,107.67),'Jawa Tengah':(-7.15,110.14),
    'Yogyakarta':(-7.87,110.43),'Jawa Timur':(-7.54,112.24),
    'Bali':(-8.34,115.09),'Nusa Tenggara Barat':(-8.65,117.36),
    'Nusa Tenggara Timur':(-8.66,121.08),'Kalimantan Barat':(0.13,111.09),
    'Kalimantan Tengah':(-1.68,113.38),'Kalimantan Selatan':(-3.09,115.28),
    'Kalimantan Timur':(1.64,116.42),'Kalimantan Utara':(3.07,116.04),
    'Sulawesi Utara':(0.62,124.02),'Sulawesi Tengah':(-1.43,121.44),
    'Sulawesi Selatan':(-3.66,120.19),'Sulawesi Tenggara':(-4.14,122.17),
    'Gorontalo':(0.54,123.06),'Sulawesi Barat':(-2.84,119.23),
    'Maluku':(-3.24,130.14),'Maluku Utara':(1.57,127.81),
    'Papua Barat':(-1.33,133.17),'Papua':(-4.27,138.08),
}

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.875rem 0;border-bottom:1px solid #111827;margin-bottom:1.5rem;'>
        <div style='font-family:"Playfair Display",serif;font-size:.95rem;
                    font-weight:600;color:#e5e7eb;'>Mobility Explorer</div>
        <div style='font-size:.6rem;color:#374151;margin-top:.3rem;
                    text-transform:uppercase;letter-spacing:.1em;'>
            P2 · Human Mobility Intelligence</div>
    </div>""", unsafe_allow_html=True)

    all_prov     = sorted(sp['provinsi'].unique())
    sel_prov     = st.multiselect("Sorot Provinsi", all_prov, default=[], placeholder="Semua...")
    mob_thr      = st.slider("Filter Mobilitas", -0.35, 0.15, (-0.35, 0.15), 0.01, format="%.2f")
    n_top        = st.slider("Jumlah Kab Ditampilkan", 5, 20, 12)

    st.markdown("""
    <div style='margin-top:1.5rem;font-size:.65rem;color:#1f2937;line-height:1.9;'>
        <div style='color:#374151;text-transform:uppercase;letter-spacing:.08em;
                    font-size:.58rem;margin-bottom:.4rem;'>Sumber Data</div>
        BPS SP2020 Long Form<br>Meta Movement Range Maps<br>
        BPS SUSENAS / SMPTK 2024<br>BPS Sensus 1980–2015
        <div style='margin-top:.75rem;color:#374151;text-transform:uppercase;
                    letter-spacing:.08em;font-size:.58rem;'>Repo</div>
        <a href='https://github.com/mikoo97/human-mobility-id'
           style='color:#3b82f6;text-decoration:none;'>
           github.com/mikoo97/human-mobility-id</a>
    </div>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Indonesia Mobility<br><em>Flow Explorer</em></div>
    <p class="hero-sub">
        Migrasi struktural &amp; mobilitas harian &nbsp;&middot;&nbsp; 1980–2024
        &nbsp;&middot;&nbsp; 34 provinsi &nbsp;&middot;&nbsp; 482 kabupaten/kota
    </p>
</div>""", unsafe_allow_html=True)

# ── KPI ──────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
kpis = [
    (c1,'Total Migran Masuk', f"{sp['migrasi_masuk'].sum()/1e6:.2f}M",
     'SP2020 · 34 provinsi','blue',False),
    (c2,'Total Migran Keluar',f"{sp['migrasi_keluar'].sum()/1e6:.2f}M",
     'SP2020 · 34 provinsi','red',False),
    (c3,'Mobilitas COVID',    f"{magg['mob_mean'].mean()*100:.1f}%",
     'vs baseline · 482 kab','amber',True),
    (c4,'Provinsi Penerima', f"{(sp['migrasi_neto']>0).sum()}/34",
     'neto masuk positif','green',False),
]
for col,lbl,val,sub,cls,neg in kpis:
    with col:
        nc = ' neg' if neg else ''
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value{nc}">{val}</div>
            <div class="kpi-sub">{sub}</div>
            <div class="kpi-rule {cls}"></div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
t1,t2,t3,t4 = st.tabs([
    "Migrasi Struktural",
    "Mobilitas COVID-19",
    "Korelasi & Insight",
    "Peta Mobilitas",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — MIGRASI STRUKTURAL
# ═══════════════════════════════════════════════════════════════
with t1:
    df_s = sp.sort_values('migrasi_neto', ascending=True).copy()
    if sel_prov:
        clrs = ['#f59e0b' if p in sel_prov
                else ('#10b981' if v>=0 else '#ef4444')
                for p,v in zip(df_s['provinsi'],df_s['migrasi_neto'])]
    else:
        clrs = ['#10b981' if v>=0 else '#ef4444' for v in df_s['migrasi_neto']]

    ca,cb = st.columns([3,2])

    with ca:
        st.markdown('<div class="sec">Migrasi Neto per Provinsi — SP2020</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=df_s['migrasi_neto']/1000, y=df_s['provinsi'],
            orientation='h', marker_color=clrs, marker_opacity=.8,
            customdata=np.stack([df_s['migrasi_masuk']/1000,
                                 df_s['migrasi_keluar']/1000,
                                 df_s['migrasi_neto']/1000],axis=-1),
            hovertemplate='<b>%{y}</b><br>Masuk: %{customdata[0]:,.1f}k<br>'
                          'Keluar: %{customdata[1]:,.1f}k<br>'
                          'Neto: %{customdata[2]:+,.1f}k<extra></extra>',
        ))
        fig.add_vline(x=0, line_dash='dot', line_color='#374151', line_width=1)
        theme(fig, h=640, xaxis_title='Migrasi Neto (ribuan jiwa)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<div class="sec">Masuk vs Keluar</div>', unsafe_allow_html=True)
        fig2 = px.scatter(sp, x='migrasi_keluar', y='migrasi_masuk',
                          color='migrasi_neto', color_continuous_scale='RdYlGn',
                          hover_name='provinsi',
                          labels={'migrasi_keluar':'Migrasi Keluar (jiwa)',
                                  'migrasi_masuk':'Migrasi Masuk (jiwa)',
                                  'migrasi_neto':'Migrasi Neto'})
        mx = max(sp['migrasi_masuk'].max(), sp['migrasi_keluar'].max())
        fig2.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',
                                  line=dict(color='#374151',dash='dot',width=1),
                                  showlegend=False,hoverinfo='skip'))
        fig2.update_traces(marker=dict(size=9,opacity=.8,
                                       line=dict(width=1.5,color='#080c12')),
                           selector=dict(type='scatter',mode='markers'))
        theme(fig2, h=280, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sec" style="margin-top:.5rem;">Top 5</div>', unsafe_allow_html=True)
        t5r = sp.nlargest(5,'migrasi_neto')
        t5k = sp.nsmallest(5,'migrasi_neto')
        fig3 = make_subplots(rows=1,cols=2,
                             horizontal_spacing=0.25)
        fig3.add_trace(go.Bar(x=t5r['migrasi_neto']/1000,y=t5r['provinsi'],
                              orientation='h',marker_color='#10b981',marker_opacity=.8,
                              showlegend=False,
                              hovertemplate='%{y}: %{x:+.0f}k<extra></extra>'),row=1,col=1)
        fig3.add_trace(go.Bar(x=t5k['migrasi_neto']/1000,y=t5k['provinsi'],
                              orientation='h',marker_color='#ef4444',marker_opacity=.8,
                              showlegend=False,
                              hovertemplate='%{y}: %{x:+.0f}k<extra></extra>'),row=1,col=2)
        theme(fig3, h=230, margin=dict(l=0,r=10,t=30,b=0))
        fig3.add_annotation(x=0.22, y=1.08, text='Penerima Terbesar',
                            xref='paper', yref='paper', showarrow=False,
                            font=dict(size=10, color='#4b5563'))
        fig3.add_annotation(x=0.85, y=1.08, text='Pengirim Terbesar',
                            xref='paper', yref='paper', showarrow=False,
                            font=dict(size=10, color='#4b5563'))
        fig3.update_annotations(font_color='#4b5563',font_size=10)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="insight">
        DKI Jakarta menjadi pengirim migran terbesar (neto -500rb+),
        mencerminkan suburbanisasi ke Jawa Barat dan Jawa Tengah. Paradoks:
        Jawa Tengah yang dikenal sebagai daerah pengirim TKI justru menjadi
        penerima neto terbesar dalam SP2020.
    </div>""", unsafe_allow_html=True)

    # Komuter
    st.markdown('<div class="sec" style="margin-top:1.5rem;">Pekerja Komuter per Provinsi — 2024</div>',
                unsafe_allow_html=True)
    df_k = kom.sort_values('komuter_perdesaan_total',ascending=False).head(20)
    fig_k = go.Figure(go.Bar(
        x=df_k['komuter_perdesaan_total']/1000, y=df_k['provinsi'],
        orientation='h', marker_color='#f59e0b', marker_opacity=.8,
        customdata=df_k['pct_perdesaan'],
        hovertemplate='<b>%{y}</b><br>%{x:.0f}k komuter<br>% perdesaan: %{customdata:.1f}%<extra></extra>'
    ))
    theme(fig_k, h=440, xaxis_title='Pekerja Komuter (ribuan)', showlegend=False)
    st.plotly_chart(fig_k, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — MOBILITAS COVID
# ═══════════════════════════════════════════════════════════════
with t2:
    df_mf = magg[(magg['mob_mean']>=mob_thr[0]) & (magg['mob_mean']<=mob_thr[1])].copy()

    ca2,cb2 = st.columns([2,1])
    with ca2:
        st.markdown('<div class="sec">Tren Mobilitas Nasional 2021–2022</div>', unsafe_allow_html=True)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=mbln['date'], y=mbln['movement_change']*100,
            mode='lines+markers',
            line=dict(color='#3b82f6',width=2),
            marker=dict(size=6,color='#3b82f6',line=dict(color='#080c12',width=1.5)),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.06)',
            hovertemplate='%{x|%b %Y}<br>%{y:.1f}%<extra></extra>'
        ))
        fig_ts.add_hline(y=0,line_dash='dot',line_color='#374151',line_width=1)
        for d,lbl,ay in [('2021-07-01','PPKM Darurat',-18),
                          ('2022-02-01','Omicron',-30),('2021-10-01','Recovery',5)]:
            idx = (mbln['date']-pd.to_datetime(d)).abs().idxmin()
            yv  = mbln.loc[idx,'movement_change']*100
            fig_ts.add_annotation(x=d,y=yv,text=lbl,showarrow=True,
                arrowhead=2,arrowcolor='#ef4444',arrowsize=.7,ay=ay,ax=0,
                font=dict(size=9,color='#ef4444'),
                bgcolor='rgba(8,12,18,.9)',bordercolor='#ef4444',borderwidth=1)
        theme(fig_ts,h=300,yaxis_title='Perubahan Mobilitas (%)',showlegend=False)
        st.plotly_chart(fig_ts, use_container_width=True)

    with cb2:
        st.markdown('<div class="sec">Statistik</div>', unsafe_allow_html=True)
        s = magg['mob_mean'].describe()
        for lbl,val,clr in [
            ('Kabupaten/Kota', f"{int(s['count'])}", '#3b82f6'),
            ('Rata-rata',      f"{s['mean']*100:.1f}%", '#f59e0b'),
            ('Std Deviasi',    f"{s['std']*100:.1f}%",  '#6b7280'),
            ('Terendah',       f"{s['min']*100:.1f}%",  '#ef4444'),
            ('Tertinggi',      f"{s['max']*100:.1f}%",  '#10b981'),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:.5rem 0;border-bottom:1px solid #111827;'>
                <span style='font-size:.75rem;color:#4b5563;'>{lbl}</span>
                <span style='font-family:"Playfair Display",serif;
                             font-weight:600;color:{clr};font-size:.95rem;'>{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc,cd = st.columns(2)
    with cc:
        st.markdown(f'<div class="sec">Top {n_top} Paling Terdampak</div>', unsafe_allow_html=True)
        td = df_mf.nsmallest(n_top,'mob_mean')
        fig_d = go.Figure(go.Bar(x=td['mob_mean']*100, y=td['polygon_name'],
                                 orientation='h', marker_color='#ef4444', marker_opacity=.8,
                                 hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'))
        fig_d.update_yaxes(categoryorder='total ascending')
        theme(fig_d,h=400,xaxis_title='Perubahan Mobilitas (%)')
        st.plotly_chart(fig_d, use_container_width=True)

    with cd:
        st.markdown(f'<div class="sec">Top {n_top} Paling Resilient</div>', unsafe_allow_html=True)
        tu = df_mf.nlargest(n_top,'mob_mean')
        fig_u = go.Figure(go.Bar(x=tu['mob_mean']*100, y=tu['polygon_name'],
                                 orientation='h', marker_color='#10b981', marker_opacity=.8,
                                 hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'))
        fig_u.update_yaxes(categoryorder='total descending')
        theme(fig_u,h=400,xaxis_title='Perubahan Mobilitas (%)')
        st.plotly_chart(fig_u, use_container_width=True)

    st.markdown('<div class="sec" style="margin-top:1rem;">Distribusi Mobilitas</div>', unsafe_allow_html=True)
    data = magg['mob_mean'].dropna()
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=data*100,nbinsx=45,
                                  marker_color='#3b82f6',marker_opacity=.65,
                                  marker_line=dict(color='#080c12',width=.4),
                                  hovertemplate='%{x:.1f}% — %{y} kab<extra></extra>'))
    fig_h.add_vline(x=data.mean()*100,line_dash='dot',line_color='#f59e0b',
                    annotation_text=f"Mean {data.mean()*100:.1f}%",
                    annotation_font_color='#f59e0b',annotation_position='top right')
    theme(fig_h,h=220,xaxis_title='Perubahan Mobilitas (%)',
          yaxis_title='Kab/Kota',showlegend=False,bargap=.06)
    st.plotly_chart(fig_h, use_container_width=True)

    # Tren historis
    st.markdown('<hr style="border-color:#1e2533;margin:2rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Tren Migrasi Historis 1980–2020 (BPS)</div>', unsafe_allow_html=True)

    # Build tren absolut
    df_t19 = sus19[['provinsi','migran_masuk_absolut']].copy()
    df_t19['tahun'] = 2019
    df_t19 = df_t19.rename(columns={'migran_masuk_absolut':'migrasi_masuk'})
    df_t19['migrasi_neto'] = np.nan
    df_t19['provinsi'] = df_t19['provinsi'].replace(NORM)

    df_sp20 = sp[['provinsi','migrasi_masuk','migrasi_neto']].copy()
    df_sp20['tahun'] = 2020
    df_sp20['provinsi'] = df_sp20['provinsi'].replace(NORM)

    df_tren = pd.concat([
        hist[['provinsi','tahun','migrasi_masuk','migrasi_neto']],
        df_t19[['provinsi','tahun','migrasi_masuk','migrasi_neto']],
        df_sp20[['provinsi','tahun','migrasi_masuk','migrasi_neto']],
    ], ignore_index=True).sort_values(['provinsi','tahun'])

    ce,cf = st.columns([1,3])
    with ce:
        prov_opts = sorted(df_tren['provinsi'].dropna().unique())
        defaults  = [p for p in ['Jakarta Raya','Jawa Tengah','Jawa Timur'] if p in prov_opts]
        prov_sel  = st.multiselect("Pilih provinsi:", prov_opts, default=defaults, key='tren_prov')
        metric    = st.radio("Metrik:", ["Masuk","Neto"], key='tren_metric')

    with cf:
        if prov_sel:
            vcol = 'migrasi_masuk' if metric == 'Masuk' else 'migrasi_neto'
            df_p = df_tren[df_tren['provinsi'].isin(prov_sel)].dropna(subset=[vcol])
            clrseq = px.colors.qualitative.Set2
            fig_tr = go.Figure()
            for ci,prov in enumerate(prov_sel):
                dp = df_p[df_p['provinsi']==prov].sort_values('tahun')
                c  = clrseq[ci % len(clrseq)]
                fig_tr.add_trace(go.Scatter(
                    x=dp['tahun'], y=dp[vcol]/1000,
                    mode='lines+markers', name=prov,
                    line=dict(color=c,width=2),
                    marker=dict(size=7,color=c,line=dict(width=1.5,color='#080c12')),
                    hovertemplate=f'<b>{prov}</b><br>%{{x}}: %{{y:,.1f}}k<extra></extra>'
                ))
            fig_tr.add_vrect(x0=2020.5,x1=2022.5,
                             fillcolor='rgba(255,255,255,0.02)',layer='below',line_width=0,
                             annotation_text='Gap 2021–22',
                             annotation_position='top left',
                             annotation_font=dict(size=8,color='#374151'))
            theme(fig_tr,h=360,
                  xaxis=dict(**AX,title='Tahun'),
                  yaxis=dict(**AX,title=f'Migrasi {metric} (ribuan jiwa)'),
                  legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7280'),
                              orientation='h',y=1.08))
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Pilih minimal 1 provinsi.")

    st.markdown("""
    <div class="insight">
        Sumber: BPS Sensus 1980–2010, SUPAS 1985–2015, SUSENAS 2019, SP2020.
        Area abu-abu = periode tanpa data publik (2021–2022).
    </div>""", unsafe_allow_html=True)

    # Komparasi 2021 vs 2022
    st.markdown('<hr style="border-color:#1e2533;margin:2rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Komparasi Mobilitas 2021 vs 2022 per Provinsi</div>',
                unsafe_allow_html=True)

    @st.cache_data
    def build_prov_yr(_magg):
        if _magg['provinsi'].isna().all():
            return None
        df = (_magg.dropna(subset=['provinsi'])
                   .copy())
        # Need year-level data - load from original
        return None  # will load separately

    @st.cache_data
    def load_meta_yr():
        base2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(base2,'data','processed',
                                      'meta_mobility_agregat_kabupaten.csv'))
        spatial = os.path.join(os.path.dirname(base2),
                               'spatial-base-system','data','spatial','processed')
        gpkg = os.path.join(spatial,'idn_kabupaten_2023_gadm.gpkg')
        if not os.path.exists(gpkg):
            return None
        import geopandas as gpd
        gdf = gpd.read_file(gpkg)[['nama_kab','nama_prov']]
        gdf['key'] = gdf['nama_kab'].str.strip().str.title()
        df['key']  = df['polygon_name'].str.strip().str.title()
        df = df.merge(gdf[['key','nama_prov']],on='key',how='left')
        prov = (df.dropna(subset=['nama_prov'])
                  .groupby(['nama_prov','tahun'])['movement_change_mean']
                  .mean().reset_index()
                  .rename(columns={'nama_prov':'provinsi','movement_change_mean':'mob'}))
        prov['mob_pct'] = (prov['mob']*100).round(2)
        return prov

    prov_yr = load_meta_yr()
    if prov_yr is not None:
        pv = prov_yr.pivot(index='provinsi',columns='tahun',values='mob_pct').reset_index()
        pv.columns.name = None
        if 2021 in pv.columns and 2022 in pv.columns:
            pv['delta'] = pv[2022] - pv[2021]
            pv = pv.sort_values('delta')

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name='2021',x=pv['provinsi'],y=pv[2021],
                                     marker_color='#3b82f6',marker_opacity=.75,
                                     hovertemplate='<b>%{x}</b><br>2021: %{y:.1f}%<extra></extra>'))
            fig_cmp.add_trace(go.Bar(name='2022',x=pv['provinsi'],y=pv[2022],
                                     marker_color='#ef4444',marker_opacity=.75,
                                     hovertemplate='<b>%{x}</b><br>2022: %{y:.1f}%<extra></extra>'))
            fig_cmp.add_hline(y=0,line_dash='dot',line_color='#374151',line_width=1)
            theme(fig_cmp,h=360,barmode='group',
                  xaxis=dict(**AX,tickangle=-40),
                  yaxis=dict(**AX,title='Perubahan Mobilitas (%)'),
                  legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7280'),
                              orientation='h',y=1.06))
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown('<div class="sec" style="margin-top:1rem;">Delta 2021 → 2022</div>',
                        unsafe_allow_html=True)
            dc = ['#10b981' if v>=0 else '#ef4444' for v in pv['delta']]
            fig_dt = go.Figure(go.Bar(x=pv['provinsi'],y=pv['delta'],
                                      marker_color=dc,marker_opacity=.8,
                                      hovertemplate='<b>%{x}</b><br>Delta: %{y:+.2f}pp<extra></extra>'))
            fig_dt.add_hline(y=0,line_dash='dot',line_color='#374151',line_width=1)
            theme(fig_dt,h=280,
                  xaxis=dict(**AX,tickangle=-40),
                  yaxis=dict(**AX,title='Delta (pp)'))
            st.plotly_chart(fig_dt, use_container_width=True)

            st.markdown("""
            <div class="insight warn">
                Delta positif = mobilitas membaik 2021 ke 2022.
                Delta negatif = makin turun di 2022, kemungkinan dampak gelombang Omicron Feb 2022.
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Geodata P1 tidak tersedia untuk agregasi tahunan.")

# ═══════════════════════════════════════════════════════════════
# TAB 3 — KORELASI
# ═══════════════════════════════════════════════════════════════
with t3:
    if len(jn) >= 5:
        x = jn['migrasi_neto'] / 1000
        y = jn['meta_mean'] * 100
        slope,intercept,r,p,_ = scipy_stats.linregress(x, y)

        ca3,cb3 = st.columns([3,2])
        with ca3:
            st.markdown('<div class="sec">Korelasi Migrasi Neto vs Mobilitas COVID</div>',
                        unsafe_allow_html=True)
            fig_sc = px.scatter(jn, x='migrasi_neto', y='meta_mean',
                                hover_name='provinsi',
                                color='meta_mean', color_continuous_scale='RdYlBu',
                                labels={'migrasi_neto':'Migrasi Neto SP2020 (jiwa)',
                                        'meta_mean':'Perubahan Mobilitas (rasio)'})
            xr = np.linspace(x.min(), x.max(), 100)
            fig_sc.add_trace(go.Scatter(x=xr*1000, y=(slope*xr+intercept)/100,
                                         mode='lines',
                                         line=dict(color='#ef4444',dash='dot',width=1.5),
                                         showlegend=False, hoverinfo='skip'))
            fig_sc.update_traces(marker=dict(size=11,opacity=.8,
                                              line=dict(color='#080c12',width=1.5)),
                                  selector=dict(type='scatter',mode='markers'))
            theme(fig_sc, h=420, coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)

        with cb3:
            st.markdown('<div class="sec">Hasil Statistik</div>', unsafe_allow_html=True)
            sig   = p < 0.05
            bc    = '#10b981' if sig else '#ef4444'
            bbg   = 'rgba(16,185,129,.08)' if sig else 'rgba(239,68,68,.08)'
            bbd   = 'rgba(16,185,129,.25)' if sig else 'rgba(239,68,68,.25)'
            btxt  = 'Signifikan (p<0.05)' if sig else 'Tidak Signifikan (p>0.05)'

            for lbl,val in [('Pearson r',f'{r:.4f}'),
                             ('r²',f'{r**2:.4f}'),('p-value',f'{p:.4f}')]:
                st.markdown(f"""
                <div style='margin-bottom:1.25rem;'>
                    <div style='font-size:.6rem;color:#374151;text-transform:uppercase;
                                letter-spacing:.1em;margin-bottom:.3rem;'>{lbl}</div>
                    <div style='font-family:"Playfair Display",serif;font-size:1.75rem;
                                font-weight:600;color:#60a5fa;'>{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='padding:.75rem 1rem;border-radius:3px;
                        background:{bbg};border:1px solid {bbd};
                        border-left:2px solid {bc};'>
                <div style='font-size:.8rem;font-weight:600;color:{bc};'>{btxt}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="insight" style="margin-top:1rem;">
                Migrasi neto tidak berkorelasi signifikan.
                Volume arus (masuk+keluar) lebih berkorelasi —
                provinsi high-turnover lebih terdampak COVID.
            </div>""", unsafe_allow_html=True)

        # Heatmap
        st.markdown('<div class="sec" style="margin-top:1.5rem;">Correlation Matrix</div>',
                    unsafe_allow_html=True)
        df_cm = jn[['migrasi_masuk','migrasi_keluar','migrasi_neto','meta_mean']].copy()
        df_cm.columns = ['Mig. Masuk','Mig. Keluar','Mig. Neto','Mobilitas COVID']
        cm = df_cm.corr().round(3)
        fig_hm = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.columns.tolist(),
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            text=cm.values.round(3), texttemplate='<b>%{text}</b>',
            textfont=dict(size=13,color='white'),
            hovertemplate='%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>',
            colorbar=dict(tickfont=dict(color='#6b7280'),len=.8,
                          title=dict(text='r',font=dict(color='#6b7280')))
        ))
        theme(fig_hm, h=320)
        fig_hm.update_xaxes(side='bottom', tickangle=-15)
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("""
        <div class="insight ok">
            Migrasi Keluar berkorelasi dengan Mobilitas COVID (r=-0.61) — provinsi
            banyak mengirim migran justru mobilitas lebih turun saat pandemi.
            Konsisten dengan teori ketergantungan remitansi.
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Data korelasi tidak cukup.")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — PETA MOBILITAS
# ═══════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="sec">Peta Choropleth per Provinsi</div>', unsafe_allow_html=True)

    @st.cache_data
    def load_geojson():
        import geopandas as gpd
        base2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sp_dir = os.path.join(os.path.dirname(base2),
                              'spatial-base-system','data','spatial','processed')
        gpkg   = os.path.join(sp_dir,'idn_provinsi_2023_gadm.gpkg')
        if not os.path.exists(gpkg):
            return None
        gdf = gpd.read_file(gpkg)
        gdf['geometry'] = gdf['geometry'].simplify(.01, preserve_topology=True)
        gdf = gdf.to_crs('EPSG:4326')
        gdf['id'] = gdf['nama_prov']
        geo = json.loads(gdf.to_json())
        for f in geo['features']:
            f['id'] = f['properties']['nama_prov']
        return geo

    geo = load_geojson()

    df_map = meta_prov.copy() if len(meta_prov) > 0 else pd.DataFrame()
    if len(df_map) == 0 and len(jn) > 0:
        df_map = jn[['provinsi','meta_mean']].copy()

    if geo is not None and len(df_map) > 0:
        sp_fix = sp.copy()
        sp_fix['provinsi'] = sp_fix['provinsi'].replace(NORM)
        df_m2 = df_map.merge(sp_fix[['provinsi','migrasi_neto',
                                      'migrasi_masuk','migrasi_keluar']],
                             on='provinsi', how='left')
        df_m2['mob_pct']   = (df_m2['meta_mean']*100).round(2)
        df_m2['neto_ribu'] = (df_m2['migrasi_neto']/1000).round(1)

        ce4,_ = st.columns([3,1])
        with ce4:
            map_m = st.radio("Warna berdasarkan:",
                             ["Mobilitas COVID (%)","Migrasi Neto (ribu jiwa)"],
                             horizontal=True)

        ccol   = 'mob_pct' if 'Mobilitas' in map_m else 'neto_ribu'
        cscale = 'RdYlBu'  if 'Mobilitas' in map_m else 'RdYlGn'
        crange = [-32,15]  if 'Mobilitas' in map_m else \
                 [df_m2['neto_ribu'].min(), df_m2['neto_ribu'].max()]

        fig_map = px.choropleth_mapbox(
            df_m2, geojson=geo, locations='provinsi', color=ccol,
            color_continuous_scale=cscale, range_color=crange,
            mapbox_style='carto-darkmatter', zoom=3.8,
            center={'lat':-2.5,'lon':118}, opacity=.72,
            hover_name='provinsi',
            hover_data={'mob_pct':':.1f','neto_ribu':':+.1f',ccol:False},
            labels={'mob_pct':'Mobilitas COVID (%)','neto_ribu':'Migrasi Neto (rb)','provinsi':'Provinsi'},
        )
        fig_map.update_layout(
            paper_bgcolor=BG, height=540,
            margin=dict(l=0,r=0,t=0,b=0),
            coloraxis_colorbar=dict(
                title=dict(text=map_m,font=dict(color='#6b7280',size=10)),
                tickfont=dict(color='#6b7280'),
                len=.6, thickness=12, x=1.01,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        worst = df_m2.nsmallest(1,'mob_pct')['provinsi'].values[0]
        best  = df_m2.nlargest(1,'mob_pct')['provinsi'].values[0]
        wv    = df_m2.nsmallest(1,'mob_pct')['mob_pct'].values[0]
        bv    = df_m2.nlargest(1,'mob_pct')['mob_pct'].values[0]
        st.markdown(f"""
        <div class="insight">
            Peta choropleth berdasarkan <b>{map_m}</b>.<br>
            Paling terdampak: <b>{worst}</b> ({wv:.1f}%) &nbsp;·&nbsp;
            Paling resilient: <b>{best}</b> ({bv:.1f}%)<br>
            Geometri: GADM v4.1 via P1 Spatial Base System.
        </div>""", unsafe_allow_html=True)

        with st.expander("Lihat data lengkap per provinsi"):
            df_show = df_m2[['provinsi','mob_pct','neto_ribu']].copy()
            df_show.columns = ['Provinsi','Mobilitas COVID (%)','Migrasi Neto (rb)']
            df_show = df_show.sort_values('Mobilitas COVID (%)')
            st.dataframe(df_show, use_container_width=True, hide_index=True)
    elif geo is None:
        st.warning("Geodata P1 tidak ditemukan.")
    else:
        st.warning("Data mobilitas tidak tersedia.")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:4rem;padding:1.25rem 0;border-top:1px solid #111827;
            text-align:center;color:#1f2937;font-size:.62rem;letter-spacing:.06em;'>
    P2 &nbsp;&middot;&nbsp; Human Mobility Intelligence &nbsp;&middot;&nbsp;
    Spatial Economic Intelligence Indonesia &nbsp;&middot;&nbsp;
    <a href='https://github.com/mikoo97/human-mobility-id'
       style='color:#3b82f6;text-decoration:none;'>GitHub</a>
</div>""", unsafe_allow_html=True)
