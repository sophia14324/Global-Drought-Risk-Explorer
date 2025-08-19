# Tracking drought risk across East Africa

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sophia14324-tracking-drought-risk-across-srcdashboardapp-navgit.streamlit.app/)

[![Live Demo](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://sophia14324-tracking-drought-risk-across-srcdashboardapp-navgit.streamlit.app/)

## 🚀 Live Demo
[Open the dashboard on Streamlit »](https://sophia14324-tracking-drought-risk-across-srcdashboardapp-navgit.streamlit.app/)

<!-- Optional: add a screenshot to assets/readme_preview.png -->
<!-- ![Dashboard preview](assets/readme_preview.png) -->

### Command-line workflow
python -m src.data.era5         
python -m src.data.gee          

python -m src.data.preprocess   
python -m src.modelling.risk_index
python -m src.modelling.clustering
streamlit run src/dashboard/app.py


#### edit config.py if needed (date range, region, grid, etc.)
