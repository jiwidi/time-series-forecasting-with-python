from tqdm import tqdm
import requests
from pandas import read_csv
from urllib.request import urlopen
import os
import logging

#Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    req = requests.get(url, stream=True)
    file_size = int(req.headers['Content-length'])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    
    with(open(dst, 'wb')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    
    pbar.close()
    return file_size

urls = {"air_pollution": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"} #Air pollution dataset, https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        

    
for name,url in urls.items():
    logger.info(f"Downloading dataset {name}")
    download_from_url(url,f"{name}.csv")
    

logger.info(f"Processing air pollution dataset")
#Proccess air pollution dataset
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('air_pollution.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# save to file
dataset.to_csv('air_pollution.csv')
logger.info(f"Processing done")