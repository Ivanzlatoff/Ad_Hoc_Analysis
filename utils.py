import plotly.express as px
import streamlit as st

# Centralised function for generating pie charts
def create_pie_chart(data, values, names, title):
  return px.pie(
      data,
      values=values,
      names=names,
      title=title,
      color_discrete_sequence=px.colors.sequential.Plasma,  # Specify color sequence
  )