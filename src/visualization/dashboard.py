"""
GB BESS Market Dashboard

Run with:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GB BESS Market Dashboard",
    page_icon="\u26a1",
    layout="wide",
)

RAW = Path(__file__).parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_auctions():
    path = RAW / "auction_results_2023-01-01_2023-10-31.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["EFA Date", "Delivery Start", "Delivery End"])


@st.cache_data
def load_system_prices():
    path = RAW / "system_prices_2023-07-01_2023-10-31.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["settlementDate", "startTime"])


@st.cache_data
def load_market_index():
    path = RAW / "market_index_2023-07-01_2023-10-31.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["settlementDate", "startTime"])


@st.cache_data
def load_generation():
    path = RAW / "generation_by_fuel_2023-07-01_2023-10-31.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(
        path, parse_dates=["settlementDate", "startTime", "publishTime"]
    )


auctions = load_auctions()
sys_prices = load_system_prices()
mkt_index = load_market_index()
gen_fuel = load_generation()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

if not auctions.empty:
    all_services = sorted(auctions["Service"].unique())
    selected_services = st.sidebar.multiselect(
        "DC/DR/DM Services", all_services, default=all_services
    )
    auction_filtered = auctions[auctions["Service"].isin(selected_services)]

    date_min = auctions["EFA Date"].min().date()
    date_max = auctions["EFA Date"].max().date()
    date_range = st.sidebar.date_input(
        "Auction date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
    )
    if len(date_range) == 2:
        auction_filtered = auction_filtered[
            (auction_filtered["EFA Date"].dt.date >= date_range[0])
            & (auction_filtered["EFA Date"].dt.date <= date_range[1])
        ]
else:
    auction_filtered = auctions
    selected_services = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("GB BESS Market Dashboard")
st.markdown("Data from the **Elexon Insights Solution API** and **NESO Data Portal**.")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Auction Records", f"{len(auction_filtered):,}")
col2.metric("System Price Records", f"{len(sys_prices):,}")
col3.metric("Market Index Records", f"{len(mkt_index):,}")
col4.metric("Generation Records", f"{len(gen_fuel):,}")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_auction, tab_system, tab_gen, tab_cross = st.tabs(
    ["DC/DR/DM Auctions", "System Prices", "Generation Mix", "Cross-Source"]
)

# ---------------------------------------------------------------------------
# Tab 1: Auctions
# ---------------------------------------------------------------------------
with tab_auction:
    if auction_filtered.empty:
        st.warning("No auction data loaded. Run the data collector first.")
    else:
        st.subheader("Clearing Prices Over Time")
        fig = px.line(
            auction_filtered,
            x="EFA Date",
            y="Clearing Price",
            color="Service",
            labels={"Clearing Price": "£/MW/h"},
        )
        fig.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Price Distribution by Service")
            fig = px.box(
                auction_filtered,
                x="Service",
                y="Clearing Price",
                color="Service",
                labels={"Clearing Price": "£/MW/h"},
            )
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Price by EFA Block")
            fig = px.box(
                auction_filtered,
                x="EFA",
                y="Clearing Price",
                color="Service",
                labels={
                    "Clearing Price": "£/MW/h",
                    "EFA": "EFA Block",
                },
            )
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Statistics")
        stats = (
            auction_filtered.groupby("Service")
            .agg(
                avg_price=("Clearing Price", "mean"),
                median_price=("Clearing Price", "median"),
                max_price=("Clearing Price", "max"),
                avg_volume=("Cleared Volume", "mean"),
                records=("Clearing Price", "count"),
            )
            .round(2)
        )
        st.dataframe(stats, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: System Prices
# ---------------------------------------------------------------------------
with tab_system:
    if sys_prices.empty:
        st.warning("No system price data loaded.")
    else:
        st.subheader("Daily System Prices")
        daily_sp = (
            sys_prices.groupby("settlementDate")
            .agg(
                avg_ssp=("systemSellPrice", "mean"),
                avg_sbp=("systemBuyPrice", "mean"),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_ssp"],
                name="Avg SSP",
                line=dict(color="#EF553B"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_sbp"],
                name="Avg SBP",
                line=dict(color="#636EFA"),
            )
        )
        fig.update_layout(
            height=450,
            template="plotly_white",
            yaxis_title="£/MWh",
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Intraday Profile (by Settlement Period)")
            sp_profile = (
                sys_prices.groupby("settlementPeriod")["systemSellPrice"]
                .agg(["mean", "median"])
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sp_profile["settlementPeriod"],
                    y=sp_profile["mean"],
                    name="Mean",
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sp_profile["settlementPeriod"],
                    y=sp_profile["median"],
                    name="Median",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Settlement Period",
                yaxis_title="£/MWh",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("SBP – SSP Spread Distribution")
            spread = sys_prices["systemBuyPrice"] - sys_prices["systemSellPrice"]
            fig = px.histogram(spread, nbins=80, labels={"value": "£/MWh"})
            fig.update_layout(
                height=400, template="plotly_white", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Generation Mix
# ---------------------------------------------------------------------------
with tab_gen:
    if gen_fuel.empty:
        st.warning("No generation data loaded.")
    else:
        st.subheader("Daily Generation by Fuel Type")
        daily_gen = (
            gen_fuel.groupby(["settlementDate", "fuelType"])["generation"]
            .sum()
            .reset_index()
        )
        gen_pivot = daily_gen.pivot_table(
            index="settlementDate",
            columns="fuelType",
            values="generation",
            fill_value=0,
        )
        col_order = gen_pivot.sum().sort_values(ascending=False).index
        gen_pivot = gen_pivot[col_order]

        melted = gen_pivot.reset_index().melt(
            id_vars="settlementDate",
            var_name="Fuel Type",
            value_name="Generation",
        )
        fig = px.area(
            melted,
            x="settlementDate",
            y="Generation",
            color="Fuel Type",
            labels={"Generation": "MW", "settlementDate": "Date"},
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Share by Fuel Type")
        fuel_share = gen_pivot.mean()
        fuel_share = fuel_share[fuel_share > 0].sort_values(ascending=False)
        fig = px.pie(values=fuel_share.values, names=fuel_share.index)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Cross-source
# ---------------------------------------------------------------------------
with tab_cross:
    if sys_prices.empty or auctions.empty:
        st.warning("Need both system price and auction data for cross-source analysis.")
    else:
        st.subheader("System Price vs DC High Clearing Price")

        daily_sys = (
            sys_prices.groupby("settlementDate")["systemSellPrice"]
            .mean()
            .reset_index()
        )
        daily_sys.columns = ["date", "avg_system_price"]

        dc_high = auctions[auctions["Service"] == "DCH"]
        daily_dc = (
            dc_high.groupby("EFA Date")["Clearing Price"].mean().reset_index()
        )
        daily_dc.columns = ["date", "avg_dc_clearing_price"]

        merged = pd.merge(daily_sys, daily_dc, on="date", how="inner")

        if merged.empty:
            st.info("No overlapping dates between system prices and DC auctions.")
        else:
            corr = merged[["avg_system_price", "avg_dc_clearing_price"]].corr().iloc[0, 1]
            st.metric("Pearson Correlation", f"{corr:.3f}")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["avg_system_price"],
                    name="Avg SSP",
                    line=dict(color="#EF553B"),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["avg_dc_clearing_price"],
                    name="Avg DC High",
                    line=dict(color="#636EFA"),
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="System Price (£/MWh)", secondary_y=False)
            fig.update_yaxes(title_text="DC Clearing (£/MW/h)", secondary_y=True)
            fig.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Scatter: Correlation")
            fig = px.scatter(
                merged,
                x="avg_system_price",
                y="avg_dc_clearing_price",
                trendline="ols",
                labels={
                    "avg_system_price": "Avg SSP (£/MWh)",
                    "avg_dc_clearing_price": "Avg DC High (£/MW/h)",
                },
                title=f"r = {corr:.3f}",
            )
            fig.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
