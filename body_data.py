from constants import KEYPOINT_DICT
import pandas as pd

class PartData:

    def __init__(self, name):
        self.name = name
        self._x = []
        self._y = []
        self._confidence = []
    
    def x(self):
        return self._x
    
    def y(self):
        return self._y
    
    def confidence(self):
        return self._confidence

    def add_coordinate(self, x, y, conf):
        self._x.append(x)
        self._y.append(y)
        self._confidence.append(conf)

    def get_data(self):
        return self.x(), self.y(), self.confidence()

class BodyData:

    def __init__(self):
        self.body_data = {}
        self.is_empty = True
        for key, _ in KEYPOINT_DICT.items():
            self.body_data[key] = PartData(key)
    
    def add_datapoint(self, keypoints_with_scores):
        if self.is_empty:
            self.is_empty = False
        
        for key, item in self.body_data.items():
            y, x, conf = keypoints_with_scores[0][0][KEYPOINT_DICT[key]]
            item.add_coordinate(x, y, conf)
    
    def to_frame(self):
        if self.is_empty:
            return
        
        x_dict, y_dict, c_dict, all_data = {}, {}, {}, {}
        for key, item in self.body_data.items():
            x_dict[key], y_dict[key], c_dict[key] = item.x(), item.y(), item.confidence()
            all_data[f"x_{key}"] = item.x()
            all_data[f"y_{key}"] = item.y()
            all_data[f"c_{key}"] = item.confidence()
        
        x_frame, y_frame, c_frame, body_data_df = pd.DataFrame(x_dict), pd.DataFrame(y_dict), pd.DataFrame(c_dict), pd.DataFrame(all_data)

        return x_frame, y_frame, c_frame, body_data_df