import bentoml
import numpy as np
from bentoml.io import NumpyNdarray 

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()
svc = bentoml.Service("homework_1", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result
