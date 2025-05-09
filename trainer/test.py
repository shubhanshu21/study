import kagglehub

# Download latest version
path = kagglehub.dataset_download("debashis74017/stock-market-data-nifty-100-stocks-5-min-data")

print("Path to dataset files:", path)