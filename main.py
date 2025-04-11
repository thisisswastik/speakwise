from app.ui import launch_app
import os
os.environ["STREAMLIT_WATCH_SUPPORT"] = "false"
import streamlit as st
from app.ui import launch_app


if __name__ == "__main__":
    launch_app()