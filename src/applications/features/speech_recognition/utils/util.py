import config_with_yaml as config
import os
from typing import Dict
import torch
from .model import enhance_pipeline


cfg = config.load(os.path.join(os.path.dirname(__file__), "config.yaml"))

if cfg.getProperty("device"):
    DEVICE = cfg.getProperty("device")
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = enhance_pipeline(cfg.getProperty("model_name"))


def export_xml(data: Dict):
    xmlcontents = '<?xml version="1.0" encoding="utf-8" ?><timedtext format="3">\n<body>'
    for text in data["chunks"]:
        time_start, time_end = text["timestamp"]
        time_start *= 1000
        time_end *= 1000
        xmlcontents += '<p t="{1}" d="{2}">{0}</p>\n'.format(
            text["text"].lower(), time_start, time_end-time_start)
    xmlcontents += '</body>\n</timedtext>'

    return xmlcontents
