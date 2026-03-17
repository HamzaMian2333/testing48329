import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.title("Small Business Demand Forecasting")
st.write("Upload a CSV file, use a template or map your columns, and generate a forecast with business insights.")


# -----------------------------
# Helpers
# -----------------------------
def clean_sales_column(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("USD", "", regex=False)
        .str.replace("usd", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def find_matching_column(df_columns, possible_names):
    lower_map = {col.lower().strip(): col for col in df_columns}
    for name in possible_names:
        if name.lower().strip() in lower_map:
            return lower_map[name.lower().strip()]
    return None


def detect_template_columns(df: pd.DataFrame, template_name: str):
    cols = list(df.columns)

    templates = {
        "Shopify Orders Export": {
            "date_candidates": ["Created at", "created at", "Date", "date", "Processed at", "processed at"],
            "sales_candidates": ["Total", "total", "Net sales", "net sales", "Subtotal", "subtotal"]
        },
        "Square Sales Export": {
            "date_candidates": ["Date", "date", "Time", "time", "Created At", "created at"],
            "sales_candidates": ["Total Collected", "total collected", "Gross Sales", "gross sales", "Amount", "amount", "Net Total", "net total"]
        },
        "Walmart Weekly Sales": {
            "date_candidates": ["Date", "date"],
            "sales_candidates": ["Weekly_Sales", "weekly_sales", "Weekly Sales"]
        }
    }

    if template_name not in templates:
        return None, None

    date_col = find_matching_column(cols, templates[template_name]["date_candidates"])
    sales_col = find_matching_column(cols, templates[template_name]["sales_candidates"])
    return date_col, sales_col


def clean_and_prepare_data(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.DataFrame:
    work_df = df[[date_col, sales_col]].copy()
    work_df.rename(columns={date_col: "date", sales_col: "sales"}, inplace=True)

    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df["sales"] = clean_sales_column(work_df["sales"])

    work_df = work_df.dropna(subset=["date", "sales"])

    work_df = (
        work_df.groupby(work_df["date"].dt.date)["sales"]
        .sum()
        .reset_index()
    )

    work_df["date"] = pd.to_datetime(work_df["date"])

    full_dates = pd.date_range(work_df["date"].min(), work_df["date"].max(), freq="D")
    work_df = work_df.set_index("date").reindex(full_dates, fill_value=0).reset_index()
    work_df.columns = ["date", "sales"]

    prophet_df = work_df.rename(columns={"date": "ds", "sales": "y"}).sort_values("ds")
    return prophet_df


def get_weekday_summary(prophet_df: pd.DataFrame) -> pd.DataFrame:
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
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    return model


def make_forecast(model: Prophet, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_insights(future_forecast: pd.DataFrame, weekday_summary: pd.DataFrame, historical_df: pd.DataFrame) -> list[str]:
    insights = []

    busiest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmax(), "weekday"]
    slowest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmin(), "weekday"]

    highest_forecast_row = future_forecast.loc[future_forecast["yhat"].idxmax()]
    lowest_forecast_row = future_forecast.loc[future_forecast["yhat"].idxmin()]

    historical_avg = historical_df["y"].mean()
    future_avg = future_forecast["yhat"].mean()

    if future_avg > historical_avg:
        insights.append("Sales appear to be trending upward compared to your historical average.")
    elif future_avg < historical_avg:
        insights.append("Sales appear to be trending slightly below your historical average.")
    else:
        insights.append("Sales appear relatively stable compared to your historical average.")

    insights.append(f"Your busiest historical weekday is {busiest_day}, while {slowest_day} is typically your slowest.")
    insights.append(
        f"Your highest forecasted day is {highest_forecast_row['ds'].date()} "
        f"with expected sales of {highest_forecast_row['yhat']:.2f}."
    )
    insights.append(
        f"Your lowest forecasted day is {lowest_forecast_row['ds'].date()} "
        f"with expected sales of {lowest_forecast_row['yhat']:.2f}."
    )

    if busiest_day in ["Friday", "Saturday", "Sunday"]:
        insights.append("Consider increasing staffing or inventory ahead of the weekend.")
    else:
        insights.append("Demand is strongest on weekdays, so plan staffing around your weekday peaks.")

    return insights


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
forecast_days = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14)

business_type = st.sidebar.selectbox(
    "Business type",
    ["Restaurant", "Retail", "Salon", "Gym", "Other"]
)

template_type = st.sidebar.selectbox(
    "Import template",
    ["Custom / Auto Detect", "Shopify Orders Export", "Square Sales Export", "Walmart Weekly Sales"]
)

st.sidebar.markdown("---")

st.sidebar.header("How to Use This App")

st.sidebar.markdown(
"""
**Step 1 — Upload your CSV**

Export your sales data from Shopify, Square, Excel, or your POS system and upload it.

---

**Step 2 — Choose a Template**

Select a template that matches your export:
- Shopify Orders Export
- Square Sales Export
- Custom / Auto Detect

The app will try to automatically find your date and sales columns.

---

**Step 3 — Confirm Columns**

Make sure the correct columns are selected:
- **Date column** → when the sale happened  
- **Sales column** → revenue or total sales

---

**Step 4 — Click Generate Forecast**

The app will analyze your historical sales and predict the next **7–30 days** of demand.

---

**Step 5 — Read the Insights**

The dashboard will show:

• expected sales for upcoming days  
• busiest weekday  
• slowest weekday  
• recommendations for staffing or inventory
"""
)

st.sidebar.markdown("---")

st.sidebar.header("How to Read the Forecast")

st.sidebar.markdown(
"""
**yhat**

This is the **predicted sales value** for that day.

Example:  
If yhat = 500, the model expects about $500 in sales.

---

**yhat_lower**

This is the **lower estimate**.

Sales are unlikely to go below this number.

---

**yhat_upper**

This is the **upper estimate**.

Sales are unlikely to go above this number.

---

**Example**

If the forecast shows:

yhat = 500  
yhat_lower = 420  
yhat_upper = 580  

This means the model expects around $500, but sales could reasonably fall between $420 and $580.

---

**Busiest Weekday**

The day that historically generates the **highest average sales**.

---

**Slowest Weekday**

The day that historically has the **lowest average sales**.

---

**Recommendations**

These suggestions help you plan:
- staffing
- inventory orders
- busy days vs slow days
"""
)

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
    st.dataframe(raw_df.head(), use_container_width=True)

    columns = list(raw_df.columns)

    auto_date_col = None
    auto_sales_col = None

    if template_type != "Custom / Auto Detect":
        auto_date_col, auto_sales_col = detect_template_columns(raw_df, template_type)

        if auto_date_col and auto_sales_col:
            st.success(f"Template matched: date = '{auto_date_col}', sales = '{auto_sales_col}'")
        else:
            st.warning("Template could not confidently match both columns. Please select them manually below.")

    default_date_index = 0
    default_sales_index = 0

    if auto_date_col in columns:
        default_date_index = columns.index(auto_date_col)

    if auto_sales_col in columns:
        default_sales_index = columns.index(auto_sales_col)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select the date column", options=columns, index=default_date_index)
    with col2:
        sales_col = st.selectbox("Select the sales/revenue column", options=columns, index=default_sales_index)

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
            weekday_summary["avg_sales"] = weekday_summary["avg_sales"].round(2)

            busiest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmax(), "weekday"]
            slowest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmin(), "weekday"]

            highest_forecast_row = future_forecast.loc[future_forecast["yhat"].idxmax()]
            lowest_forecast_row = future_forecast.loc[future_forecast["yhat"].idxmin()]

            total_next_period = future_forecast["yhat"].sum()
            avg_daily_next_period = future_forecast["yhat"].mean()

            st.subheader("Key Insights")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Forecast Days", forecast_days)
            k2.metric("Expected Sales", f"{total_next_period:,.2f}")
            k3.metric("Avg Daily Sales", f"{avg_daily_next_period:,.2f}")
            k4.metric("Busiest Weekday", busiest_day)

            k5, k6, k7 = st.columns(3)
            k5.metric("Slowest Weekday", slowest_day)
            k6.metric("Highest Forecast Day", f"{highest_forecast_row['ds'].date()}")
            k7.metric("Lowest Forecast Day", f"{lowest_forecast_row['ds'].date()}")

            st.subheader("Recommendations")
            insights = build_insights(future_forecast, weekday_summary, prophet_df)

            if business_type == "Restaurant":
                st.info("\n\n".join(insights + [
                    "For restaurants, use this to plan staffing, food prep, and ordering."
                ]))
            elif business_type == "Retail":
                st.info("\n\n".join(insights + [
                    "For retail, use this to plan purchasing, shelf inventory, and peak-day coverage."
                ]))
            elif business_type == "Salon":
                st.info("\n\n".join(insights + [
                    "For salons, use this to anticipate appointment load and staffing needs."
                ]))
            elif business_type == "Gym":
                st.info("\n\n".join(insights + [
                    "For gyms, use this to anticipate class and foot-traffic demand."
                ]))
            else:
                st.info("\n\n".join(insights))

            st.subheader(f"Next {forecast_days} Days Forecast")
            st.dataframe(future_forecast, use_container_width=True)

            csv_data = convert_df_to_csv(future_forecast)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name="forecast_output.csv",
                mime="text/csv"
            )

            st.subheader("Forecast Chart")
            fig1 = model.plot(forecast)
            plt.title("Sales Forecast")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            st.pyplot(fig1)

            st.subheader("Trend and Seasonality")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            st.subheader("Average Historical Sales by Weekday")
            st.dataframe(weekday_summary, use_container_width=True)

            st.subheader("Weekday Sales Pattern")
            fig3, ax = plt.subplots(figsize=(10, 4))
            ax.bar(weekday_summary["weekday"].astype(str), weekday_summary["avg_sales"])
            ax.set_title("Average Historical Sales by Weekday")
            ax.set_xlabel("Weekday")
            ax.set_ylabel("Average Sales")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

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

