from deepface import DeepFace
import cv2

# Tradução das emoções para português
emotion_translation = {
    "angry": "Raiva",
    "disgust": "Desgosto",
    "fear": "Medo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Surpresa",
    "neutral": "Neutro"
}

# Captura a imagem da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detecta os rostos na imagem
    faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)

    if faces:
        # Análise das emoções
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Emoção detectada e traduzida
        emotion = result[0]['dominant_emotion']
        emotion_pt = emotion_translation.get(emotion, emotion)

        # Coordenadas do rosto detectado
        x, y, w, h = result[0]['region']['x'], result[0]['region']['y'], result[0]['region']['w'], result[0]['region']['h']
        
        # Desenha o retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Exibe a emoção detectada em português
        cv2.putText(frame, emotion_pt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Exibe a imagem com a emoção detectada
    cv2.imshow('Webcam', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
