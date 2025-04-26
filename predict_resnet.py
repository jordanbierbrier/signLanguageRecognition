import cv2
import numpy as np
import time
import os
import tensorflow as tf
from keras.models import load_model

# Mapping from model output labels to ASL letters
LABEL_TO_LETTER = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm',
    12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's',
    18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'
}


def load_trained_model(model_path='models/resnet.h5'):
    """Load the pretrained ResNet model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return load_model(model_path)


def background_removal(frame, back_sub, gaussian=True):
    """Remove background and extract hand region."""
    roi = frame[100:500, 100:400]
    foreground = back_sub.apply(roi, learningRate=0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    if gaussian:
        foreground = cv2.GaussianBlur(foreground, (3, 3), 0)

    hand = cv2.bitwise_and(roi, roi, mask=foreground)
    return hand


def preprocess_for_resnet(image):
    """Prepare segmented hand image for ResNet prediction."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_3_channel = np.repeat(gray_img[..., np.newaxis], 3, axis=-1)
    img_3_channel = np.expand_dims(img_3_channel, axis=0)
    return img_3_channel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    model = load_trained_model()
    prev_frame_time = 0
    capture_background = False
    prev_letter = ''
    word = ''
    confidence = 0
    display_word = True
    back_sub = None

    print("Press 'b' to capture background and start prediction.")
    print("Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rect_frame = cv2.rectangle(frame, (100, 100), (400, 500), (255, 0, 0), 3)

        if capture_background and back_sub is not None:
            hand_img = background_removal(frame, back_sub)

            if np.mean(cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)) > 10:
                input_img = preprocess_for_resnet(hand_img)
                preds = model.predict(input_img, verbose=0)
                pred_letter = LABEL_TO_LETTER[np.argmax(preds)]

                cv2.putText(rect_frame, "Guessed Letter:", (500, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(rect_frame, pred_letter, (800, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

                if pred_letter == prev_letter:
                    confidence += 1
                else:
                    confidence = 0

                if confidence > 10 and display_word:
                    word += pred_letter
                    confidence = 0

                prev_letter = pred_letter

            # Overlay hand segmentation into frame
            rect_frame[100:500, 100:400] = hand_img

        if display_word:
            cv2.putText(rect_frame, word, (100, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time + 1e-5)
        prev_frame_time = new_frame_time
        cv2.putText(rect_frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Sign Language Recognition (ResNet)", rect_frame)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('b'):
            capture_background = True
            back_sub = cv2.createBackgroundSubtractorMOG2(0, 100)
        elif key == ord(' '):
            word += ' '
        elif key == ord('s'):
            display_word = not display_word
        elif key == 127:  # Backspace
            word = word[:-1]
        elif key == ord('r'):
            word = ''

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()