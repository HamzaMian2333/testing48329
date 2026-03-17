import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.title("Small Business Demand Forecasting")
st.write("Upload a CSV file, map your date and sales columns, and generate a forecast.")

# -----------------------------
# Helpers
# -----------------------------
def clean_and_prepare_data(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.DataFrame:
    """
    Clean uploaded data and return a daily aggregated dataframe
    with Prophet-friendly columns: ds, y
    """
    work_df = df.copy()

    work_df = work_df[[date_col, sales_col]].copy()
    work_df.columns = ["date", "sales"]

    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df["sales"] = pd.to_numeric(work_df["sales"], errors="coerce")

    work_df = work_df.dropna(subset=["date", "sales"])

    # Aggregate to daily totals
    work_df = (
        work_df.groupby(work_df["date"].dt.date, as_index=False)["sales"]
        .sum()
        .rename(columns={"date": "date"})
    )
    work_df["date"] = pd.to_datetime(work_df["date"])

    # Fill missing dates with 0 sales so the series is continuous
    full_dates = pd.date_range(work_df["date"].min(), work_df["date"].max(), freq="D")
    work_df = work_df.set_index("date").reindex(full_dates, fill_value=0).reset_index()
    work_df.columns = ["date", "sales"]

    prophet_df = work_df.rename(columns={"date": "ds", "sales": "y"}).sort_values("ds")
    return prophet_df


def get_weekday_summary(prophet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average sales by weekday based on historical data.
    """
    weekday_df = prophet_df.copy()
    weekday_df["weekday"] = weekday_df["ds"].dt.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    summary = (
        weekday_df.groupby("weekday", as_index=False)["y"]
        .mean()
        .rename(columns={"y": "avg_sales"})
    )

    summary["weekday"] = pd.Categorical(summary["weekday"], categories=weekday_order, ordered=True)
    summary = summary.sort_values("weekday").reset_index(drop=True)
    return summary


def train_model(prophet_df: pd.DataFrame) -> Prophet:
    """
    Train Prophet model.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    return model


def make_forecast(model: Prophet, periods: int) -> pd.DataFrame:
    """
    Forecast future periods.
    """
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    if raw_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    columns = list(raw_df.columns)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select the date column", options=columns)
    with col2:
        sales_col = st.selectbox("Select the sales column", options=columns)

    if st.button("Generate Forecast"):
        try:
            prophet_df = clean_and_prepare_data(raw_df, date_col, sales_col)

            if len(prophet_df) < 14:
                st.warning("This dataset has fewer than 14 days of usable data. Forecast quality may be weak.")

            if prophet_df["y"].sum() == 0:
                st.error("All cleaned sales values are zero. Please check your selected sales column.")
                st.stop()

            model = train_model(prophet_df)
            forecast = make_forecast(model, forecast_days)

            future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days).copy()
            future_forecast["yhat"] = future_forecast["yhat"].round(2)
            future_forecast["yhat_lower"] = future_forecast["yhat_lower"].round(2)
            future_forecast["yhat_upper"] = future_forecast["yhat_upper"].round(2)

            weekday_summary = get_weekday_summary(prophet_df)
            busiest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmax(), "weekday"]
            slowest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmin(), "weekday"]

            # -----------------------------
            # KPI cards
            # -----------------------------
            st.subheader("Key Insights")

            total_next_period = future_forecast["yhat"].sum()
            avg_daily_next_period = future_forecast["yhat"].mean()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Forecast Days", forecast_days)
            k2.metric("Expected Sales", f"{total_next_period:,.2f}")
            k3.metric("Avg Daily Sales", f"{avg_daily_next_period:,.2f}")
            k4.metric("Busiest Day", busiest_day)

            st.write(f"**Slowest Day:** {slowest_day}")

            # -----------------------------
            # Forecast table
            # -----------------------------
            st.subheader(f"Next {forecast_days} Days Forecast")
            st.dataframe(future_forecast, use_container_width=True)

            csv_data = convert_df_to_csv(future_forecast)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name="forecast_output.csv",
                mime="text/csv"
            )

            # -----------------------------
            # Forecast chart
            # -----------------------------
            st.subheader("Forecast Chart")
            fig1 = model.plot(forecast)
            plt.title("Sales Forecast")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            st.pyplot(fig1)

            # -----------------------------
            # Seasonal components
            # -----------------------------
            st.subheader("Trend and Seasonality")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # -----------------------------
            # Weekday summary
            # -----------------------------
            st.subheader("Average Historical Sales by Weekday")
            weekday_display = weekday_summary.copy()
            weekday_display["avg_sales"] = weekday_display["avg_sales"].round(2)
            st.dataframe(weekday_display, use_container_width=True)

            # -----------------------------
            # Historical data preview after cleaning
            # -----------------------------
            st.subheader("Cleaned Historical Daily Data")
            cleaned_display = prophet_df.rename(columns={"ds": "date", "y": "sales"}).copy()
            st.dataframe(cleaned_display.tail(30), use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the forecast: {e}")

else:
    st.info("Upload a CSV to get started.")
    st.markdown(
    """
Example CSV format:

```csv
date,sales
2025-01-01,420
2025-01-02,390
2025-01-03,510
"""
    )

