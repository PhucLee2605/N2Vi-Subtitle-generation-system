from utils import metric, preprocess


data = preprocess.handlejsonData("captionData.json")
print(metric.valuateTranslate('envit5', data))