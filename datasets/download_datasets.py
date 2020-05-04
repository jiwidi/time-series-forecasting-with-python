from tqdm import tqdm
from pandas import read_csv
from datetime import datetime

import numpy as np
import logging
import requests
import io
import os

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    req = requests.get(url, stream=True, verify=False)
    file_size = int(req.headers["Content-length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    pbar = tqdm(
        total=file_size,
        initial=first_byte,
        unit="B",
        unit_scale=True,
        desc=url.split("/")[-1],
    )

    with (open(dst, "wb")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)

    pbar.close()
    return file_size


def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")


if __name__ == "__main__":
    # Air pollution dataset,
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    urls = {
        "air_pollution": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
    }

    # Air pollution
    # =================
    logger.info(f"Processing air pollution dataset")
    x = requests.get(url=urls["air_pollution"], verify=False).content
    dataset = read_csv(
        io.StringIO(x.decode("utf8")),
        parse_dates=[["year", "month", "day", "hour"]],
        index_col=0,
        date_parser=parse,
    )
    dataset.drop("No", axis=1, inplace=True)
    dataset.columns = [
        "pollution_today",
        "dew",
        "temp",
        "press",
        "wnd_dir",
        "wnd_spd",
        "snow",
        "rain",
    ]
    dataset.index.name = "date"
    # Drop first 24 hours because they have incomplete data
    dataset = dataset[24:]
    # Fix na values with interpolation, our analysis just detected 36 days out
    # of 1200 with nans but we are not that confident about doing this
    dataset["pollution_today"].fillna(0, inplace=True)
    dataset.pollution_today = dataset.pollution_today.replace(
        0, np.nan).interpolate()
    # Dataset had hourly measurements, we will translate this into the avg
    # pollution for the day and 25%/75% percentlies. Losing categorical
    # feature wind dir
    dataset = dataset.resample("D").mean()
    dataset = dataset[dataset.pollution_today != 0]
    #     #Snow and rain columns where binary columns initially but now when resampling to daily data they have values greater than 1, lets fix it
    #     dataset['snow'] = (dataset.snow > 0).astype(int)
    #     dataset['rain'] = (dataset.rain > 0).astype(int)
    # Make it to supervised learning, every raw of datapoints at time t has
    # target t+1
    dataset["pollution_yesterday"] = (
        dataset["pollution_today"].tolist()[-1:]
        + dataset["pollution_today"].tolist()[:-1]
    )
    # save to file
    path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"Saving air pollution dataset to{path}")
    dataset.to_csv(f"{path}/air_pollution.csv")

    logger.info(f"Air pollution dataset processed")
    logger.info(f"Processing done")
