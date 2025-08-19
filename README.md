# Tracking drought risk across East Africa

[![Live Demo](https://sophia14324-tracking-drought-risk-across-srcdashboardapp-navgit.streamlit.app/)

### Command-line workflow
python -m src.data.era5         # downloads ERA5-Land netCDF
python -m src.data.gee          # exports monthly NDVI & LST to Drive → data/

python -m src.data.preprocess   # builds indicators
python -m src.modelling.risk_index
python -m src.modelling.clustering
streamlit run src/dashboard/app.py


#### edit config.py if needed (date range, region, grid, etc.)
