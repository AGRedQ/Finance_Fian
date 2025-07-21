import streamlit as st
import json           


# FrontendFian will provide visualization, streamlit will just be streamlit...

# Too bad, making streamlit a class is not a good idea

class FrontendFian:
    def __init__(self):
        pass

    def visualize_indicator(self, data, indicator_name, title=None):
        from Fian.fian_utils import visualize_indicator
        visualize_indicator(data, indicator_name, title)


    def compare_stocks(self, data1, data2, title=None):
        from Fian.fian_utils import compare_stocks
        compare_stocks(data1, data2, title)

    def display_stock(self, data, ticker):
        from Fian.fian_utils import display_stock_info
        display_stock_info(data, ticker)