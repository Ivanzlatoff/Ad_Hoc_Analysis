import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
from PIL import Image

# Set up Streamlit configuration
st.set_page_config(page_title="Ad-Hoc Analysis", layout="wide")
st.header("Gateway Ad-Hoc Analysis")
st.subheader("Analysing Ad-Hoc MAWB's")

# Load Excel file
excel_file = "Gateway_PB_Prior_Month.xlsx"
sheet_names = pd.ExcelFile(excel_file).sheet_names

# Read all sheets into a dictionary
df = pd.read_excel(excel_file, sheet_name="Ad_Hoc", usecols="A:N", header=0).dropna()

# --- STREAMLIT SIDEBAR ---
with st.sidebar:
    st.markdown("### Filters for All Sheets")

    all_departments = pd(df["HAWB Department ID"]).unique().tolist()
    all_chargeable_weights = pd(df["HAWB Chargeable Kgs"]).unique().tolist()
    all_house_destinations = pd(df["House Destination"]).unique().tolist()
    all_hawbs = pd(df["SHP_HAWB/HBL"]).unique().tolist()

    chargeable_weight_selection = st.slider(
        "Chargeable Weight:",
        min_value=min(all_chargeable_weights),
        max_value=max(all_chargeable_weights),
        value=(min(all_chargeable_weights), max(all_chargeable_weights)),
    )

    department_selection = st.multiselect("Department:", all_departments, default=all_departments)
    house_destinations_selection = st.multiselect("House Destination:", all_house_destinations, default=all_house_destinations)

# Filter data for all sheets based on selection
filtered_df = {}
charts = []

for df in df.items():
    mask = (
        (df["HAWB Chargeable Kgs"].between(*chargeable_weight_selection))
        & (df["HAWB Department ID"].isin(department_selection))
        & (df["House Destination"].isin(house_destinations_selection))
    )
    filtered_df = df[mask]
    number_of_results = filtered_df.shape[0]
    st.markdown(f"*Available Results: {number_of_results}*")

    # Aggregate the data to count HAWBs per department
    department_counts = filtered_df.groupby("HAWB Department ID").size().reset_index(name="Number of HAWB's")

    # --- PIE CHARTS ---
    pie_chart_weight = px.pie(
        filtered_df,
        values="HAWB Chargeable Kgs",
        names="HAWB Department ID",
        title=f"Weight Distribution",
        color_discrete_sequence=px.colors.sequential.Plasma  # Specify color sequence
    )

    pie_chart_hawb = px.pie(
        department_counts,  # Use the aggregated data
        values="Number of HAWBs",
        names="HAWB Department ID",
        title=f"HAWB Distribution",
        color_discrete_sequence=px.colors.sequential.Plasma  # Specify color sequence
    )

    # Plot the charts
    col1, col2 = st.columns(2)
    col1.plotly_chart(pie_chart_weight)
    col2.plotly_chart(pie_chart_hawb)

    # --- BAR CHARTS ---
    bar_chart_destinations_hawb = px.bar(
        x="House Destination", y="Number of HAWBs",
        title=f"Number of HAWB's for each destination",
        color="Number of HAWBs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )
    bar_chart_destinations_weight = px.bar(
        x="House Destination", y="HAWB Chargeable Kgs",
        title=f"Total chargeable weight of shipments for each destination",
        color="HAWB Chargeable Kgs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )

    st.plotly_chart(bar_chart_destinations_hawb)
    st.plotly_chart(bar_chart_destinations_weight)

    # --- TOP 10 DESTINATIONS BAR CHARTS ---
    top_10_destinations_hawb = filtered_df.groupby("House Destination").size().nlargest(10).reset_index(name="Number of HAWBs")
    top_10_destinations_weight = filtered_df.groupby("House Destination")["HAWB Chargeable Kgs"].sum().reset_index().nlargest(10, "HAWB Chargeable Kgs")

    bar_chart_top_10_destinations_hawb = px.bar(
        top_10_destinations_hawb,
        x="House Destination", y="Number of HAWBs",
        title=f"Top 10 Destinations by Number of MAWB's",
        color="Number of HAWBs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )
    bar_chart_top_10_destinations_weight = px.bar(
        top_10_destinations_weight,
        x="House Destination", y="HAWB Chargeable Kgs",
        title=f"Top 10 Destinations by Total Chargeable Weight",
        color="HAWB Chargeable Kgs",  # Color by the value
        color_continuous_scale=px.colors.sequential.Plasma  # Specify color scale
    )

    st.plotly_chart(bar_chart_top_10_destinations_hawb)
    st.plotly_chart(bar_chart_top_10_destinations_weight)

    # --- TOP 10 DESTINATIONS BAR CHARTS AGGREGATED ---
    # For Chargeable Weight
    df_weight_by_dest = filtered_df.groupby(["House Destination", "HAWB Department ID"])["HAWB Chargeable Kgs"].sum().reset_index()

    # Sort by chargeable weight and get top 10 destinations
    top_10_weight_by_dest = df_weight_by_dest.groupby("House Destination").sum().nlargest(10, "HAWB Chargeable Kgs").reset_index()

    # Filter data for top 10 destinations
    filtered_weight_data = df_weight_by_dest[df_weight_by_dest["House Destination"].isin(top_10_weight_by_dest["House Destination"])]
    
    # Step 1: Sort the data by HAWB Chargeable Kgs in descending order
    filtered_weight_data_sorted = filtered_weight_data.sort_values(by="HAWB Chargeable Kgs", ascending=False)

    # Step 2: Modify the "House Destination" column to include the chargeable weight in the label
    filtered_weight_data_sorted["Destination"] = (
        filtered_weight_data_sorted["House Destination"] + 
        " (" + 
        filtered_weight_data_sorted["HAWB Chargeable Kgs"].round(2).astype(str) + 
        " kg)"
    )

    # Step 3: Create the bar chart with the sorted destinations and new labels
    bar_chart_top_10_destinations_weight_aggregated = px.bar(
        filtered_weight_data_sorted,
        x="HAWB Chargeable Kgs",
        y="HAWB Department ID",
        color="Destination",  # Use the modified label with weight
        color_discrete_sequence=px.colors.qualitative.Vivid,
        orientation="h",
        height=600,
        title=f"Top 10 Destination by Total Chargeable Weight Aggregated"
    )
    
    st.plotly_chart(bar_chart_top_10_destinations_weight_aggregated)

    # For Number of HAWBs
    df_hawb_by_dest = filtered_df.groupby(["House Destination", "HAWB Department ID"])["HAWB"].size().reset_index(name="Number of HAWBs")

    # Sort by Number of HAWBs and get top 10 destinations
    top_10_mawb_by_dest = df_hawb_by_dest.groupby("House Destination").sum().nlargest(10, "Number of HAWBs").reset_index()

    # Filter data for top 10 destinations
    filtered_mawb_data = df_hawb_by_dest[df_hawb_by_dest["House Destination"].isin(top_10_mawb_by_dest["House Destination"])]
    
    # Step 1: Sort the data by HAWB Chargeable Kgs in descending order
    filtered_mawb_data_sorted = filtered_mawb_data.sort_values(by="Number of HAWBs", ascending=False)

    # Step 2: Modify the "House Destination" column to include the number of MAWB's in the label
    filtered_mawb_data_sorted["Destination"] = (
        filtered_mawb_data_sorted["House Destination"] + " (" 
        + filtered_mawb_data_sorted["Number of HAWBs"].round(2).astype(str)
        + ")"
    )

    # Step 3: Create the bar chart with the sorted destinations and new labels
    bar_chart_top_10_destinations_hawb_aggregated = px.bar(
        filtered_mawb_data_sorted,
        x="Number of HAWBs",
        y="HAWB Department ID",
        color="Destination",  # Use the modified label with weight
        color_discrete_sequence=px.colors.qualitative.Vivid,
        orientation="h",
        height=600,
        title=f"Top 10 Destination by number of MAWB's with 1 HAWB Aggregated in {sheet_name}"
    )
    
    st.plotly_chart(bar_chart_top_10_destinations_hawb_aggregated)

    # --- TREND LINE  ---
    st.markdown(f"## Trend Line")

    # Assuming weekly data can be derived from ETD
    filtered_df['Week'] = pd.to_datetime(filtered_df['ETD']).dt.to_period('W').apply(lambda r: r.start_time)
    weekly_data = filtered_df.groupby("Week").agg(
        total_weight=pd.NamedAgg(column="HAWB Chargeable Kgs", aggfunc="sum"),
        mawb_count=pd.NamedAgg(column="MAWB", aggfunc="count")
    ).reset_index()

    # Convert Week to numerical format for linear regression
    weekly_data['Week_num'] = (weekly_data['Week'] - weekly_data['Week'].min()).dt.days

    # Linear regression model
    def add_trend_line(df, x_col, y_col):
        model = LinearRegression()
        X = df[[x_col]].values.reshape(-1, 1)
        y = df[y_col].values
        model.fit(X, y)
        trend_line = model.predict(X)
        return trend_line

    # Line chart for chargeable weight with trend line
    trend_chart_weight = go.Figure()
    trend_chart_weight.add_trace(go.Scatter(x=weekly_data["Week"], y=weekly_data["total_weight"], mode='markers+lines', name='Total Weight', line=dict(color='blue')))
    trend_chart_weight.add_trace(go.Scatter(x=weekly_data["Week"], y=add_trend_line(weekly_data, "Week_num", "total_weight"), mode='lines', name='Trend Line', line=dict(color='red', width=2)))
    trend_chart_weight.update_layout(title="Weekly Chargeable Weight with Trend Line")

    # Line chart for Number of HAWBs with trend line
    trend_chart_mawb = go.Figure()
    trend_chart_mawb.add_trace(go.Scatter(x=weekly_data["Week"], y=weekly_data["mawb_count"], mode='markers+lines', name='MAWB Count', line=dict(color='red')))
    trend_chart_mawb.add_trace(go.Scatter(x=weekly_data["Week"], y=add_trend_line(weekly_data, "Week_num", "mawb_count"), mode='lines', name='Trend Line', line=dict(color='blue', width=2)))
    trend_chart_mawb.update_layout(title="Weekly Number of HAWBs with Trend Line")

    st.plotly_chart(trend_chart_weight)
    st.plotly_chart(trend_chart_mawb)

    # Add the charts to the list for the PDF report
    charts.append((pie_chart_weight, f"Weight Distribution"))
    charts.append((pie_chart_hawb, f"HAWB Distribution"))
    charts.append((bar_chart_destinations_hawb, f"Number of HAWB's for each destination"))
    charts.append((bar_chart_destinations_weight, f"Total chargeable weight of shipments for each destination"))
    charts.append((bar_chart_top_10_destinations_hawb, f"Top 10 Destinations by Number of HAWB's"))
    charts.append((bar_chart_top_10_destinations_weight, f"Top 10 Destinations by Total Chargeable Weight"))
    charts.append((bar_chart_top_10_destinations_weight_aggregated, f"Top 10 Destination by Total Chargeable Weight Aggregated"))
    charts.append((bar_chart_top_10_destinations_hawb_aggregated, f"Top 10 Destination by number of MAWB's Aggregated"))
    charts.append((trend_chart_weight, f"Weekly Chargeable Weight with Trend Line"))
    charts.append((trend_chart_mawb, f"Weekly Number of MAWB's with Trend Line"))

# Print report button
if st.button("Print Report"):
    def save_chart_as_image(fig, filename):
        pio.write_image(fig, filename, format='png')

    def create_pdf_report(charts):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        for chart, title in charts:
            # Create a temporary file for saving the chart image
            temp_filename = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    temp_filename = tmpfile.name
                    save_chart_as_image(chart, temp_filename)
                    
                    # Open the image file using Pillow to ensure it is not locked
                    with Image.open(temp_filename) as img:
                        c.drawString(72, height - 72, title)
                        c.drawImage(temp_filename, 72, height - 600, width=width - 144, preserveAspectRatio=True)
                        c.showPage()
            finally:
                # Ensure the temporary file is deleted
                if temp_filename and os.path.isfile(temp_filename):
                    os.remove(temp_filename)

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    report = create_pdf_report(charts)
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.pdf",
        mime="application/pdf"
    )