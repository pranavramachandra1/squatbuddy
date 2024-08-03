from constants import KEYPOINT_DICT

class PartData:

    def __init__(self, name):
        self.name = name
        self._x = []
        self._y = []
    
    def x(self):
        return self._x
    
    def y(self):
        return self._y

    def add_coordinate(self, x, y):
        self._x.append(x)
        self._y.append(y)

    def get_data(self):
        return self.x(), self.y()

class BodyData:

    def __init__(self):
        self.body_data = {}
        for key, _ in KEYPOINT_DICT.items():
            self.body_data[key] = PartData(key)
    
    def add_datapoint(self, keypoints_with_scores):
        for key, item in self.body_data.items():
            x, y, _ = keypoints_with_scores[0][0][KEYPOINT_DICT[key]]
            item.add_coordinate(x, y)