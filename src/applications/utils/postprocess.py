import xml.etree.ElementTree as ET
from typing import Dict


def milliseconds_to_time(milliseconds):
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(hours, minutes, seconds, milliseconds)

def xml_to_srt(xml_data, srt_file):
    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()

    with open(srt_file, 'w', encoding="utf-8") as f:
        count = 1
        for subtitle in root.findall('.//p'):
            start = int(subtitle.get('t'))
            end = int(subtitle.get('d')) + start
            text = subtitle.text

            f.write(str(count) + '\n')
            f.write(milliseconds_to_time(start) + ' --> ' + milliseconds_to_time(end) + '\n')
            f.write(text + '\n\n')

            count += 1

def export_xml(data: Dict):
    xmlcontents = '<?xml version="1.0" encoding="utf-8" ?><timedtext format="3">\n<body>'
    for text in data["chunks"]:
        time_start, time_end = text["timestamp"]
        time_start *= 1000
        time_end *= 1000
        xmlcontents += '<p t="{1}" d="{2}">{0}</p>\n'.format(
            text["text"].lower(), int(time_start), int(time_end-time_start))
    xmlcontents += '</body>\n</timedtext>'

    return xmlcontents