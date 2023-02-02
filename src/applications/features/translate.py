from .utils import metric, preprocess


data = preprocess.handlejsonData("../../data/captionData.json")
print(metric.valuateTranslate('envit5', data))