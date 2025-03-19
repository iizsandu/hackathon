from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect_objects(self, image):
        results = self.model(image)
        objects = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = result.names[int(cls)]
                objects.append((label, (x1, y1, x2, y2)))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return objects, image

class ObjectComparator:
    def compare_objects(self, objects1, objects2):
        objects1_labels = [obj[0] for obj in objects1]
        objects2_labels = [obj[0] for obj in objects2]

        common_objects = list(set(objects1_labels) & set(objects2_labels))
        different_objects = list(set(objects1_labels) ^ set(objects2_labels))

        return common_objects, different_objects

class App:
    def __init__(self):
        self.detector = ObjectDetector()
        self.comparator = ObjectComparator()

    def run(self):
        st.title('Object Detection and Comparison')

        col1, col2 = st.columns(2)
        with col1:
            uploaded_image1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
        with col2:
            uploaded_image2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

        if uploaded_image1 and uploaded_image2:
            image1 = cv2.imdecode(np.frombuffer(uploaded_image1.read(), np.uint8), 1)
            image2 = cv2.imdecode(np.frombuffer(uploaded_image2.read(), np.uint8), 1)

            objects1, result_image1 = self.detector.detect_objects(image1)
            objects2, result_image2 = self.detector.detect_objects(image2)

            common_objects, different_objects = self.comparator.compare_objects(objects1, objects2)

            st.image([result_image1, result_image2], caption=["Image 1 - Detected Objects", "Image 2 - Detected Objects"], channels='BGR')
            st.write(f"Common Objects: {common_objects}")
            st.write(f"Different Objects: {different_objects}")

if __name__ == '__main__':
    app = App()
    app.run()
