import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Advanced COVID Health Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_data.csv", parse_dates=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df = df.sort_values("Date")

    # Derived metrics
    df["CFR"] = (df["Deaths"] / df["Confirmed"]) * 100
    df["RecoveryRate"] = (df["Recovered"] / df["Confirmed"]) * 100
    df["CasesPerMillion"] = (df["Confirmed"] / df["Population"]) * 1_000_000
    df["DeathsPerMillion"] = (df["Deaths"] / df["Population"]) * 1_000_000
    df["ActiveRatio"] = (df["Active"] / df["Confirmed"]) * 100

    # Growth
    df["DailyGrowth"] = df.groupby("Country")["Confirmed"].pct_change() * 100
    df["7DayAvg"] = df.groupby("Country")["Confirmed"].transform(lambda x: x.rolling(7).mean())

    return df

df = load_data()

# Sidebar
st.sidebar.title("🔍 Filters")
continent = st.sidebar.multiselect("Continent", df["Continent"].unique(), df["Continent"].unique())
country = st.sidebar.multiselect("Country", df["Country"].unique(), df["Country"].unique())
date_range = st.sidebar.date_input("Date Range", [df["Date"].min(), df["Date"].max()])

filtered_df = df[
    (df["Continent"].isin(continent)) &
    (df["Country"].isin(country)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# KPIs
total_cases = filtered_df["Confirmed"].sum()
total_deaths = filtered_df["Deaths"].sum()
total_recovered = filtered_df["Recovered"].sum()
active_cases = filtered_df["Active"].sum()
cfr = (total_deaths / total_cases * 100) if total_cases else 0
recovery_rate = (total_recovered / total_cases * 100) if total_cases else 0

st.title("🦠 Advanced COVID / Health Analytics Dashboard")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Confirmed Cases", f"{total_cases:,}")
k2.metric("Active Cases", f"{active_cases:,}")
k3.metric("Deaths", f"{total_deaths:,}")
k4.metric("Recovered", f"{total_recovered:,}")
k5.metric("CFR", f"{cfr:.2f}%")
k6.metric("Recovery Rate", f"{recovery_rate:.2f}%")

st.divider()

# Trend with Rolling Avg
st.subheader("📈 Confirmed Cases Trend (7-Day Avg)")
trend = filtered_df.groupby("Date")[["Confirmed", "7DayAvg"]].sum().reset_index()
st.plotly_chart(
    px.line(trend, x="Date", y=["Confirmed", "7DayAvg"]),
    use_container_width=True
)

# Risk Classification
st.subheader("🚦 Country Risk Levels")
risk_df = filtered_df.groupby("Country")[["CasesPerMillion"]].mean().reset_index()
risk_df["RiskLevel"] = pd.cut(
    risk_df["CasesPerMillion"],
    bins=[0, 1000, 5000, 20000, np.inf],
    labels=["Low", "Moderate", "High", "Critical"]
)
st.plotly_chart(
    px.bar(risk_df, x="Country", y="CasesPerMillion", color="RiskLevel"),
    use_container_width=True
)

# Death vs Recovery Matrix
st.subheader("⚖ Deaths vs Recovery Analysis")
dr = filtered_df.groupby("Country")[["Deaths", "Recovered"]].sum().reset_index()
st.plotly_chart(
    px.scatter(dr, x="Recovered", y="Deaths", size="Deaths", color="Country"),
    use_container_width=True
)

# Population Impact
st.subheader("👥 Population Impact (Per Million)")
pop = filtered_df.groupby("Country")[["CasesPerMillion", "DeathsPerMillion"]].mean().reset_index()
st.plotly_chart(
    px.bar(pop, x="Country", y=["CasesPerMillion", "DeathsPerMillion"]),
    use_container_width=True
)

# Peak Detection
st.subheader("🚨 Peak Case Detection")
peak = filtered_df.groupby("Country").apply(lambda x: x.loc[x["Confirmed"].idxmax()])[["Country", "Date", "Confirmed"]]
st.dataframe(peak.reset_index(drop=True), use_container_width=True)

# Data Table
st.subheader("📄 Detailed Health Records")
st.dataframe(filtered_df, use_container_width=True)

# Download
st.download_button(
    "⬇ Download Filtered Dataset",
    filtered_df.to_csv(index=False).encode("utf-8"),
    "covid_advanced_data.csv",
    "text/csv"
)

