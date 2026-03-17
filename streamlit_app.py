import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.title("Small Business Demand Forecasting")
st.write("Upload a CSV file, use a template or map your columns, and get a simple forecast with clear recommendations.")


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
        insights.append("Sales look a little higher than your usual average.")
    elif future_avg < historical_avg:
        insights.append("Sales look a little lower than your usual average.")
    else:
        insights.append("Sales look fairly steady compared to your usual average.")

    insights.append(f"Your busiest weekday is usually {busiest_day}.")
    insights.append(f"Your slowest weekday is usually {slowest_day}.")
    insights.append(
        f"Your strongest forecast day is {highest_forecast_row['ds'].date()} "
        f"at about {highest_forecast_row['yhat']:.0f} in sales."
    )
    insights.append(
        f"Your weakest forecast day is {lowest_forecast_row['ds'].date()} "
        f"at about {lowest_forecast_row['yhat']:.0f} in sales."
    )

    return insights


def generate_action_recommendations(future_forecast: pd.DataFrame) -> list[str]:
    actions = []

    max_day = future_forecast.loc[future_forecast["Predicted Sales"].idxmax()]
    min_day = future_forecast.loc[future_forecast["Predicted Sales"].idxmin()]
    avg_sales = future_forecast["Predicted Sales"].mean()

    if max_day["Predicted Sales"] > avg_sales * 1.2:
        actions.append(
            f"High demand expected on {max_day['Date'].date()}. Consider adding staff or stocking extra inventory."
        )

    if min_day["Predicted Sales"] < avg_sales * 0.8:
        actions.append(
            f"Lower demand expected on {min_day['Date'].date()}. Consider lighter staffing or a small promotion."
        )

    if not actions:
        actions.append("Demand looks fairly steady across this forecast period, so normal staffing and inventory levels may be enough.")

    return actions


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

avg_order_value = st.sidebar.number_input("Average order value ($)", min_value=1.0, value=20.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("How to use this app")
st.sidebar.markdown(
    """
1. Upload your CSV file.

2. Pick an import template if it matches your file.

3. Make sure the date column is correct.

4. Make sure the sales column is correct.

5. Click **Generate Forecast**.

6. Read the summary, recommendations, and forecast table.
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("How to read the results")
st.sidebar.markdown(
    """
- **Predicted Sales** = the app's best guess for that date

- **Low Estimate** = a lower likely value

- **High Estimate** = a higher likely value

- **Expected Sales** = total predicted sales for the forecast period

- **Avg Daily Sales** = average predicted sales per day

- **Estimated Orders** = predicted sales divided by your average order value
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simple example")
st.sidebar.markdown(
    """
If the forecast shows:

- **Predicted Sales:** 500  
- **Low Estimate:** 420  
- **High Estimate:** 580  

This means:

You will likely make about **500** in sales.

Most likely range: **420 to 580**.
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("Confidence")
st.sidebar.markdown(
    """
The low and high estimates show a reasonable range.

Real sales will often land somewhere inside that range, not exactly on one number every time.
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
            st.warning("The template could not find both columns. Please pick them manually below.")

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

            # Rename to plain English
            future_forecast.rename(
                columns={
                    "ds": "Date",
                    "yhat": "Predicted Sales",
                    "yhat_lower": "Low Estimate",
                    "yhat_upper": "High Estimate"
                },
                inplace=True
            )

            future_forecast["Estimated Orders"] = (
                future_forecast["Predicted Sales"] / avg_order_value
            ).round(0)

            weekday_summary = get_weekday_summary(prophet_df)
            weekday_summary["avg_sales"] = weekday_summary["avg_sales"].round(2)

            busiest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmax(), "weekday"]
            slowest_day = weekday_summary.loc[weekday_summary["avg_sales"].idxmin(), "weekday"]

            best_day = future_forecast.loc[future_forecast["Predicted Sales"].idxmax()]
            worst_day = future_forecast.loc[future_forecast["Predicted Sales"].idxmin()]

            total_next_period = future_forecast["Predicted Sales"].sum()
            avg_daily_next_period = future_forecast["Predicted Sales"].mean()
            total_estimated_orders = future_forecast["Estimated Orders"].sum()

            # Weekly summary card
            st.subheader("Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Expected Sales", f"{total_next_period:,.0f}")
            s2.metric("Avg Daily Sales", f"{avg_daily_next_period:,.0f}")
            s3.metric("Best Day", f"{best_day['Date'].date()}")
            s4.metric("Worst Day", f"{worst_day['Date'].date()}")

            s5, s6, s7 = st.columns(3)
            s5.metric("Best Day Sales", f"{best_day['Predicted Sales']:,.0f}")
            s6.metric("Worst Day Sales", f"{worst_day['Predicted Sales']:,.0f}")
            s7.metric("Estimated Orders", f"{total_estimated_orders:,.0f}")

            st.subheader("Key Insights")
            k1, k2, k3 = st.columns(3)
            k1.metric("Busiest Weekday", busiest_day)
            k2.metric("Slowest Weekday", slowest_day)
            k3.metric("Forecast Days", forecast_days)

            st.subheader("Recommendations")
            insights = build_insights(
                future_forecast.rename(
                    columns={
                        "Date": "ds",
                        "Predicted Sales": "yhat",
                        "Low Estimate": "yhat_lower",
                        "High Estimate": "yhat_upper"
                    }
                ),
                weekday_summary,
                prophet_df
            )

            actions = generate_action_recommendations(future_forecast)

            all_recommendations = insights + actions

            if business_type == "Restaurant":
                all_recommendations.append("For restaurants, use this to plan staffing, food prep, and ordering.")
            elif business_type == "Retail":
                all_recommendations.append("For retail, use this to plan purchasing, shelf stock, and peak-day coverage.")
            elif business_type == "Salon":
                all_recommendations.append("For salons, use this to plan appointment capacity and staffing.")
            elif business_type == "Gym":
                all_recommendations.append("For gyms, use this to anticipate foot traffic and class demand.")

            st.info("\n\n".join(all_recommendations))

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

            st.caption("The shaded area around the forecast line shows uncertainty. Real sales will often fall somewhere inside that range.")

            st.subheader("Trend and Seasonality")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            st.subheader("Average Historical Sales by Weekday")
            weekday_display = weekday_summary.rename(
                columns={"weekday": "Weekday", "avg_sales": "Average Sales"}
            )
            st.dataframe(weekday_display, use_container_width=True)

            st.subheader("Weekday Sales Pattern")
            fig3, ax = plt.subplots(figsize=(10, 4))
            ax.bar(weekday_display["Weekday"].astype(str), weekday_display["Average Sales"])
            ax.set_title("Average Historical Sales by Weekday")
            ax.set_xlabel("Weekday")
            ax.set_ylabel("Average Sales")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

            st.subheader("Cleaned Historical Daily Data")
            cleaned_display = prophet_df.rename(columns={"ds": "Date", "y": "Sales"}).copy()
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

