from .utils import preprocess
from .utils import metric


data = preprocess.handlejsonData("../../data/captionData.json")
print(metric.valuateTranslate('envit5', data))