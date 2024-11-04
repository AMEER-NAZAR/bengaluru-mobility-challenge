from typing import Dict
import cv2
import numpy as np
from collections import defaultdict
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Vector
from supervision.tools.detections import Detections



class LineCounter:
    def __init__(self, start: Point, end: Point, class_names: list):
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.class_names = class_names
        self.in_count: dict = {counter: 0 for counter in self.class_names}
        self.out_count: dict = {counter: 0 for counter in self.class_names}

    def update(self, detections: Detections):
        """
        Update the in_count and out_count for each class that crosses the line.

        :param detections: Detections : The detections to update the counts.
        """
        for xyxy, confidence, class_id, tracker_id in detections:
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            vehicle_class = self.class_names[class_id]
            if tracker_state:
                print("Class is",vehicle_class)
                self.in_count[vehicle_class] += 1
            else:
                self.out_count[vehicle_class] += 1

class LineCounterAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 1,
        text_color: Color = Color.blue(),
        text_scale: float = 0,
        text_offset: float = 10,
        text_padding: int = 20,
    ):
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineCounter, class_names: list) -> np.ndarray:
        """
        Annotates the frame with the in and out counts for each vehicle class.

        :param frame: np.ndarray : The image to annotate.
        :param line_counter: LineCounter : The line counter object.
        :param class_names: list : The vehicle class names.
        :return: np.ndarray : The annotated image.
        """
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )

        in_texts = [
            f"in {class_name}: {line_counter.in_count[class_name]}"
            for class_name in class_names
        ]
        out_texts = [
            f"out {class_name}: {line_counter.out_count[class_name]}"
            for class_name in class_names
        ]

        for idx, text in enumerate(in_texts + out_texts):
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )
            y_offset = idx * (text_height + self.text_padding)

            cv2.putText(
                frame,
                text,
                (80, 30 + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,0),
                2,
                cv2.LINE_AA,
            )

        return frame
