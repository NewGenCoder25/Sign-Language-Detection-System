import cv2
import flet as ft
import numpy as np
import threading
import time
import base64
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model


# Trie implementation for efficient word suggestions
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def get_suggestions(self, prefix, limit=5):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        self._dfs(node, prefix, suggestions, limit)
        return suggestions

    def _dfs(self, node, current, suggestions, limit):
        if len(suggestions) >= limit:
            return
        if node.is_end:
            suggestions.append(current)
        for char, child in sorted(node.children.items()):
            self._dfs(child, current + char, suggestions, limit)

# Global variables
global model, labels
try:
    model = load_model('models/keras_model.h5')
    labels = open('models/labels.txt', 'r').readlines()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    labels = []

# App state class to manage global variables
class AppState:
    def __init__(self):
        self.img_size = 224
        self.offset = 20
        self.text_buffer = ""
        self.current_prediction = ""
        self.confidence = 0.0
        self.running = False
        self.auto_mode = False
        self.last_prediction_time = 0
        self.auto_delay = 2.0
        self.last_prediction = ""
        self.last_accepted_prediction = ""
        self.confidence_threshold = 0.7


app_state = AppState()

# Initialize word suggestions
word_trie = Trie()
common_words = [
    "hello", "good", "morning", "evening", "thank", "you", "please", "sorry",
    "what", "how", "where", "who", "when", "name",
    "yes", "no", "help", "stop", "go", "want", "need", "more",
    "food", "water", "home", "bathroom", "sleep", "work", "school",
    "my", "your", "friend", "family", "doctor",
    "happy", "sad", "love", "like", "understand"
]

for word in common_words:
    word_trie.insert(word)


def extract_clean_label(raw_label):
    return ''.join([char for char in raw_label.strip() if char.isalpha()])


def get_suggestions(text, limit=5):
    if not text.strip():
        return []

    current_word = text.split()[-1].lower() if text.split() else ""
    return word_trie.get_suggestions(current_word, limit)


def process_frame(update_text_callback, update_image_callback):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    detector = HandDetector(maxHands=1)
    frame_counter = 0

    while app_state.running:
        success, img = cap.read()
        if not success:
            continue

        # Skip frames in auto mode to reduce CPU usage
        if app_state.auto_mode and frame_counter % 2 == 0:
            frame_counter += 1
            continue
        frame_counter += 1

        hands, img = detector.findHands(img)
        imgWhite = np.ones((app_state.img_size, app_state.img_size, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x1 = max(0, x - app_state.offset)
            y1 = max(0, y - app_state.offset)
            x2 = min(img.shape[1], x + w + app_state.offset)
            y2 = min(img.shape[0], y + h + app_state.offset)

            if x2 <= x1 or y2 <= y1:
                continue

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0:
                try:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        new_width = max(1, int(app_state.img_size / aspectRatio))
                        imgResize = cv2.resize(imgCrop, (new_width, app_state.img_size))
                        width_gap = (app_state.img_size - new_width) // 2
                        imgWhite[:, width_gap:width_gap + new_width] = imgResize
                    else:
                        new_height = max(1, int(app_state.img_size * aspectRatio))
                        imgResize = cv2.resize(imgCrop, (app_state.img_size, new_height))
                        height_gap = (app_state.img_size - new_height) // 2
                        imgWhite[height_gap:height_gap + new_height, :] = imgResize

                    imgResize = cv2.resize(imgWhite, (224, 224))
                    img_array = np.asarray(imgResize, dtype=np.float32).reshape(1, 224, 224, 3)
                    img_array = (img_array / 127.5) - 1

                    predictions = model.predict(img_array, verbose=0)
                    pred_index = np.argmax(predictions)
                    app_state.confidence = np.max(predictions)
                    app_state.current_prediction = extract_clean_label(labels[pred_index])

                    # Skip if confidence is below threshold
                    if app_state.confidence < app_state.confidence_threshold:
                        update_text_callback("???", app_state.text_buffer, app_state.confidence)
                        continue

                    # Auto-mode logic
                    if app_state.auto_mode:
                        if app_state.current_prediction != app_state.last_prediction:
                            app_state.last_prediction = app_state.current_prediction
                            app_state.last_prediction_time = time.time()
                        elif time.time() - app_state.last_prediction_time > app_state.auto_delay:
                            if app_state.current_prediction == "SPACE":
                                app_state.text_buffer += " "
                            elif app_state.current_prediction == "DEL":
                                app_state.text_buffer = app_state.text_buffer[:-1]
                            else:
                                app_state.text_buffer += app_state.current_prediction
                            app_state.last_accepted_prediction = app_state.current_prediction
                            app_state.last_prediction_time = time.time()

                    if update_text_callback:
                        update_text_callback(app_state.current_prediction, app_state.text_buffer, app_state.confidence)

                except Exception as e:
                    print("Prediction error:", e)

        # Convert frame to display in Flet only (this is the current feed being shown in the app window)
        _, im_arr = cv2.imencode('.png', img)
        im_b64 = base64.b64encode(im_arr).decode("utf-8")
        if update_image_callback:
            update_image_callback(im_b64)

    cap.release()

def main(page: ft.Page):
    page.title = "Sign Language Recognition"
    page.window_width = 900
    page.window_height = 800
    page.theme_mode = ft.ThemeMode.LIGHT
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Load model and labels
    try:
        model = load_model('models/keras_model.h5')
        labels = open('models/labels.txt', 'r').readlines()
    except Exception as e:
        page.add(ft.Text(f"Error loading model: {str(e)}", color="red"))
        return

    # UI Elements
    loading_indicator = ft.ProgressRing(visible=False)
    pred_text = ft.Text(value="Predicted: ", size=24, weight="bold")
    conf_text = ft.Text(value="Confidence: ", size=18)
    buffer_text = ft.Text(value="Accepted Text: ", size=20, weight="bold")
    suggestion_text = ft.Text(value="Suggested: —", size=16, italic=True, color=ft.colors.BLUE_800)
    suggestion_buttons = ft.Row(wrap=True, scroll="adaptive")
    mode_text = ft.Text(value="Mode: Manual", size=18, color=ft.colors.BLUE, weight="bold")
    image_display = ft.Image(width=640, height=480, border_radius=10, fit=ft.ImageFit.CONTAIN)

    # Control buttons
    start_button = ft.ElevatedButton(
        "Start Recognition",
        icon=ft.icons.PLAY_ARROW,
        on_click=lambda e: start_recognition()
    )

    accept_button = ft.ElevatedButton(
        "Accept (Enter)",
        icon=ft.icons.CHECK,
        on_click=lambda e: accept_prediction(),
        disabled=app_state.auto_mode
    )
    def update_delay(e):
        app_state.auto_delay = e.control.value

    def update_confidence_threshold(e):
        app_state.confidence_threshold = e.control.value
    # Sliders
    auto_delay_slider = ft.Slider(
        min=1, max=5,
        divisions=4,
        label="{value} sec",
        value=app_state.auto_delay,
        width=300,
        on_change=update_delay
    )

    confidence_slider = ft.Slider(
        min=0.5, max=1.0,
        divisions=10,
        label="Threshold: {value:.2f}",
        value=app_state.confidence_threshold,
        width=300,
        on_change=update_confidence_threshold
    )

    def apply_suggestion(word):
        words = app_state.text_buffer.strip().split()
        if words:
            words[-1] = word
            app_state.text_buffer = " ".join(words) + " "
        update_text(app_state.current_prediction, app_state.text_buffer, app_state.confidence)

    def update_text(pred, buffer, conf):
        pred_text.value = f"Predicted: {pred}"
        conf_text.value = f"Confidence: {conf:.2f}"
        buffer_text.value = f"Accepted Text: {buffer}"
        mode_text.value = f"Mode: {'AUTO' if app_state.auto_mode else 'MANUAL'}"
        mode_text.color = ft.colors.GREEN if app_state.auto_mode else ft.colors.BLUE

        suggestions = get_suggestions(buffer)
        suggestion_text.value = "Suggested: " + ", ".join(suggestions) if suggestions else "Suggested: —"

        suggestion_buttons.controls.clear()
        for s in suggestions:
            suggestion_buttons.controls.append(
                ft.FilledButton(
                    s,
                    on_click=lambda e, word=s: apply_suggestion(word),
                    height=40,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=5)
                    )
                )
            )

        page.update()

    def update_image(im_b64):
        image_display.src_base64 = im_b64
        page.update()

    def accept_prediction(e=None):
        if app_state.confidence > app_state.confidence_threshold:
            if app_state.current_prediction == "SPACE":
                app_state.text_buffer += " "
            elif app_state.current_prediction == "DEL":
                app_state.text_buffer = app_state.text_buffer[:-1]
            else:
                app_state.text_buffer += app_state.current_prediction
            app_state.last_accepted_prediction = app_state.current_prediction
        update_text(app_state.current_prediction, app_state.text_buffer, app_state.confidence)

    def clear_single_char(e=None):
        app_state.text_buffer = app_state.text_buffer[:-1] if app_state.text_buffer else ""
        update_text(app_state.current_prediction, app_state.text_buffer, app_state.confidence)

    def clear_all_text(e=None):
        app_state.text_buffer = ""
        app_state.last_accepted_prediction = ""
        update_text(app_state.current_prediction, app_state.text_buffer, app_state.confidence)

    def exit_app(e=None):
        app_state.running = False
        page.window_close()

    def toggle_mode(e=None):
        app_state.auto_mode = not app_state.auto_mode
        accept_button.disabled = app_state.auto_mode

        # Show mode change notification
        page.snack_bar = ft.SnackBar(
            content=ft.Text(f"Switched to {'AUTO' if app_state.auto_mode else 'MANUAL'} mode"),
            duration=1000
        )
        page.snack_bar.open = True

        update_text(app_state.current_prediction, app_state.text_buffer, app_state.confidence)





    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "Enter" and not app_state.auto_mode:
            accept_prediction()
        elif e.key == "Backspace":
            clear_single_char()
        elif e.key.lower() == "c":
            clear_all_text()
        elif e.key == "Escape":
            exit_app()
        elif e.key.lower() == "m":
            toggle_mode()
        elif e.key == "Tab" and app_state.text_buffer.strip():
            suggestions = get_suggestions(app_state.text_buffer)
            if suggestions:
                apply_suggestion(suggestions[0])

    def start_recognition():
        try:
            loading_indicator.visible = True
            page.update()

            # Only one camera access here (in process_frame)
            app_state.running = True
            start_button.disabled = True
            page.update()

            # Start the recognition in a separate thread
            threading.Thread(
                target=process_frame,
                args=(update_text, update_image),
                daemon=True
            ).start()

        except Exception as e:
            page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error: {str(e)}", color="red"),
                action="OK"
            )
            page.snack_bar.open = True
            page.update()
        finally:
            loading_indicator.visible = False
            page.update()

    page.on_keyboard_event = on_keyboard

    # Main layout
    page.add(
        ft.Column([
            ft.Row([loading_indicator, start_button], alignment="center"),
            ft.Divider(height=20),
            ft.Row([
                image_display,
                ft.Column([
                    pred_text,
                    conf_text,
                    ft.Divider(height=10),
                    buffer_text,
                    ft.Container(
                        ft.Column([
                            suggestion_text,
                            suggestion_buttons
                        ]),
                        padding=10,
                        border=ft.border.all(1, ft.colors.BLUE_100),
                        border_radius=5,
                        bgcolor=ft.colors.BLUE_50
                    ),
                    ft.Divider(height=20),
                    mode_text,
                    ft.Row([
                        ft.Text("Auto Delay:", width=100),
                        auto_delay_slider
                    ]),
                    ft.Row([
                        ft.Text("Confidence Threshold:", width=100),
                        confidence_slider
                    ]),
                    ft.Row([
                        accept_button,
                        ft.ElevatedButton(
                            "Clear Last (Backspace)",
                            icon=ft.icons.BACKSPACE,
                            on_click=clear_single_char
                        ),
                        ft.ElevatedButton(
                            "Clear All (C)",
                            icon=ft.icons.CLEAR,
                            on_click=clear_all_text
                        ),
                        ft.ElevatedButton(
                            "Toggle Mode (M)",
                            icon=ft.icons.SWITCH_LEFT,
                            on_click=toggle_mode
                        ),
                        ft.ElevatedButton(
                            "Exit (ESC)",
                            icon=ft.icons.EXIT_TO_APP,
                            on_click=exit_app
                        )
                    ], wrap=True)
                ], spacing=10)
            ], spacing=20)
        ], scroll=ft.ScrollMode.AUTO)
    )


ft.app(target=main)