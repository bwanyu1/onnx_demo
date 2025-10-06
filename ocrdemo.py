from onnxocr.onnx_paddleocr import ONNXPaddleOcr
import cv2  # OpenCV を使用して画像を読み込む

def sample():
    ocr = ONNXPaddleOcr(use_gpu=False, lang="japan")
    
    # 画像を読み込む
    img_path = "img/trafic.jpg"
    img = cv2.imread(img_path)  # OpenCV を使用して画像を読み込む
    
    if img is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {img_path}")
    
    # OCR を実行
    result = ocr.ocr(img)
    for data in result:
        for box, (text, score) in data:
            print(f"text: {text}, score: {score}")

if __name__ == "__main__":
    sample()
