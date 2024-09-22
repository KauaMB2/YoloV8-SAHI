import cv2
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

def detect_objects(image_path='1.png', weights='yolov8s.pt', save_result=True):
    # Verifica se o caminho da imagem existe
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

    # Caminho do modelo YOLOv8
    yolov8_model_path = f'models/{weights}'
    download_yolov8s_model(yolov8_model_path)
    
    # Inicializa o modelo de detecção SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device='cpu'# Definir como sendo 'cuda' para usar GPU e 'cpu' para user CPU
    )

    # Carrega a imagem
    image = cv2.imread(image_path)

    # Faz a predição
    results = get_sliced_prediction(
        image,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # Desenha caixas delimitadoras e rótulos na imagem
    for obj in results.object_prediction_list:
        box = obj.bbox
        cls = obj.category.name
        confidence = obj.score  # Aqui vamos manter o objeto PredictionScore
        # Verifique como acessar o valor de confiança
        try:
            confidence_value = confidence.value  # Acessa o valor de confiança, ajuste conforme necessário
        except AttributeError:
            confidence_value = float(confidence)  # Tentativa de conversão direta

        x1, y1, x2, y2 = int(box.minx), int(box.miny), int(box.maxx), int(box.maxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (56, 56, 255), 2)
        label = f"{cls} {confidence_value:.2f}"  # Usa o valor de confiança
        t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
        cv2.rectangle(image, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), (56, 56, 255), -1)
        cv2.putText(image, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Salva a imagem com as detecções
    if save_result:
        result_path = 'resultSAHI.png'
        cv2.imwrite(result_path, image)
        print(f"Result saved to {result_path}")

    # Mostra a imagem com as detecções
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects()
