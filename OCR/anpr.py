import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    engine="onnxruntime",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


# image_list = ["images/plate_1.png", "images/plate_2.png", "images/plate_3.png", "images/plate_4.png"]
image_path = 'images/Indian_number_plate/test/crops'
image_list = [os.path.join(image_path, img) for img in os.listdir(image_path) if img.endswith('.jpg') or img.endswith('.png')]
result = ocr.predict(image_list)
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

