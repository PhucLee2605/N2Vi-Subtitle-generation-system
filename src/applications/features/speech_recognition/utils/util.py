import config_with_yaml as config
import os
from typing import Dict


cfg = config.load(os.path.join(os.path.dirname(__file__), "config.yaml"))


def export_xml(data: Dict):
    with open(f'{cfg.getProperty("output_dir")}/somefile.xml', "w") as file:
        file.write('<?xml version="1.0" encoding="utf-8" ?><timedtext format="3">\n<body>')
        for text in data["chunk"]:
            time_start, time_end = text["timestamp"]
            time_start *= 1000
            time_end *= 1000
            xmlcontents = '<p t="{}" d="{}">{}</p>\n'.format(text["text"].lower(), time_start, time_end-time_start)
            file.write(xmlcontents)
        file.write('</body>\n</timedtext>')