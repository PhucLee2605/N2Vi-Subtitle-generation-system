import xml.etree.ElementTree as ET
from typing import Dict, List


def format_time(milliseconds: int) -> str:
    """ Convert time from millisecond into HH:MM:SS,ffff

    Args:
        milliseconds (int): time in millisecond to convert

    Returns:
        str: formated time
    """
    seconds, milliseconds = divmod(int(milliseconds), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(hours, minutes, seconds, milliseconds)


def xml_to_srt(xml_data: str, srt_file: str) -> None:
    """ Convert input from xml format into srt

    Args:
        xml_data (str): data in form of xml format
        srt_file (str): output file name for writing srt after convert
    """
    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()

    with open(srt_file, 'w', encoding="utf-8") as f:
        count = 1
        for subtitle in root.findall('.//p'):
            start = int(subtitle.get('t'))
            end = int(subtitle.get('d')) + start
            text = subtitle.text

            f.write(str(count) + '\n')
            f.write(format_time(start) + ' --> ' + format_time(end) + '\n')
            f.write(text + '\n\n')

            count += 1


def export_xml(data: Dict) -> str:
    """ format output of recognition into xml format

    Args:
        data (Dict): Input data must be the form of below python dictionary
                        {"text": "raw text from recognition",
                        "chunks": [
                                    {"text": "sentence1", "timestamps": (start, end)},
                                    {"text": "sentence2", "timestamps": (start, end)},
                                    ...
                                ]
                        }

    Returns:
        str: xml format of input data
    """
    xmlcontents = '<?xml version="1.0" encoding="utf-8" ?><timedtext format="3">\n<body>'
    for text in data["chunks"]:
        time_start, time_end = text["timestamp"]
        time_start *= 1000
        time_end *= 1000
        xmlcontents += '<p t="{1}" d="{2}">{0}</p>\n'.format(
            text["text"].lower(), int(time_start), int(time_end-time_start))
    xmlcontents += '</body>\n</timedtext>'

    return xmlcontents


#TODO complete docstring
def stack_chunks(chunks: List[dict], length=20, threshold=2000) -> List[str]:
    """_summary_

    Args:
        chunks (List[dict]): _description_
        length (int, optional): _description_. Defaults to 20.
        threshold (int, optional): _description_. Defaults to 2000.

    Returns:
        List[str]: _description_
    """
    begin_time = chunks[0]['timestamp'][0] * 1000
    line_end_time = chunks[0]['timestamp'][1] * 1000
    scripts = list()
    line = ""
    for index in range(len(chunks)):
        if chunks[index]['timestamp'][0] * 1000 - line_end_time > threshold or len(line) > length:
            scripts.append({'text': line.strip(), 'timestamp': [format_time(begin_time), format_time(line_end_time)]})
            line = chunks[index]['text'] + " "
            begin_time = chunks[index]['timestamp'][0] * 1000
            line_end_time = chunks[index]['timestamp'][1] * 1000
        else:
            line += chunks[index]['text'] + " "
            line_end_time = chunks[index]['timestamp'][1] * 1000

    return scripts