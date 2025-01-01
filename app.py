import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QTextEdit, QLabel, QHBoxLayout, 
                            QFrame, QListWidget, QProgressBar)
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QCursor
import pyperclip
from langchain.llms import Ollama
import time
from pynput import keyboard
from pynput.keyboard import Key, Controller, Listener
from queue import Queue
from threading import Lock

class TextBuffer:
    def __init__(self, max_size=1000):
        self.buffer = ""
        self.max_size = max_size
        self.lock = Lock()

    def append(self, text):
        with self.lock:
            self.buffer += text
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]

    def get(self):
        with self.lock:
            return self.buffer

    def clear(self):
        with self.lock:
            self.buffer = ""

class GenerationThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, llm, prompt):
        super().__init__()
        self.llm = llm
        self.prompt = prompt
        
    def run(self):
        response = self.llm(self.prompt)
        self.finished.emit(response)

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 180);
                border-radius: 20px;
            }
        """)
        layout = QVBoxLayout(self)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        layout.addWidget(self.progress)
        
        # Loading text
        self.label = QLabel("Generating...")
        self.label.setStyleSheet("color: white; font-size: 14px;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

class KeyboardMonitor(QThread):
    text_captured = pyqtSignal(str)
    
    def __init__(self, text_buffer):
        super().__init__()
        self.running = True
        self.text_buffer = text_buffer
        
    def run(self):
        def on_press(key):
            if not self.running:
                return False
                
            try:
                if hasattr(key, 'char') and key.char:
                    self.text_buffer.append(key.char)
                elif key == keyboard.Key.space:
                    buffer_content = self.text_buffer.get()
                    if buffer_content:
                        self.text_captured.emit(buffer_content)
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def stop(self):
        self.running = False

class SuggestionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.keyboard = Controller()
        
        layout = QVBoxLayout(self)
        self.suggestions = QListWidget()
        self.suggestions.itemClicked.connect(self.use_suggestion)
        layout.addWidget(self.suggestions)
        
        self.setStyleSheet("""
            QListWidget {
                background-color: rgba(40, 44, 52, 0.95);
                color: white;
                border-radius: 5px;
                border: 1px solid #3d3d3d;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #4a9eff;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #3d8ce4;
            }
        """)
        
    def use_suggestion(self, item):
        text = item.text()
        pyperclip.copy(text)
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(Key.ctrl)
        self.hide()

class RephraseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.keyboard = Controller()

        layout = QVBoxLayout(self)
        
        # Header with close button
        header = QHBoxLayout()
        title = QLabel("Rephrase Text")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        header.addWidget(title)
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(25, 25)
        close_btn.clicked.connect(self.hide)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ffffff;
                font-size: 18px;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #ff4455;
            }
        """)
        header.addWidget(close_btn)
        layout.addLayout(header)
        
        self.input = QTextEdit()
        self.input.setPlaceholderText("How would you like to rephrase this?")
        self.input.setMinimumWidth(300)
        self.input.setMinimumHeight(100)
        layout.addWidget(self.input)
        
        self.rephrase_btn = QPushButton("Rephrase")
        self.rephrase_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d8ce4;
            }
        """)
        self.rephrase_btn.clicked.connect(self.rephrase_text)
        layout.addWidget(self.rephrase_btn)
        
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 44, 52, 0.95);
                border-radius: 10px;
                border: 1px solid #3d3d3d;
            }
            QTextEdit {
                background-color: rgba(55, 60, 70, 0.95);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        # Loading overlay
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loading_overlay.resize(self.size())
        
    def rephrase_text(self):
        self.loading_overlay.show()
        instructions = self.input.toPlainText()
        selected_text = pyperclip.paste()
        
        # Create generation thread
        self.gen_thread = GenerationThread(
            self.parent().llm,
            f"Rephrase the following text: {selected_text}\nInstructions: {instructions}"
        )
        self.gen_thread.finished.connect(self.handle_rephrased_text)
        self.gen_thread.start()
    
    def handle_rephrased_text(self, rephrased):
        pyperclip.copy(rephrased)
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(Key.ctrl)
        self.loading_overlay.hide()
        self.hide()

class FloatingAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Writing Assistant")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.keyboard = Controller()
        
        # Initialize Ollama
        self.llm = Ollama(model="hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q5_K_S")  # Use your preferred model
        
        # Initialize text buffer
        self.text_buffer = TextBuffer(max_size=2000)  # Stores last 2000 characters
        
        # Initialize widgets
        self.suggestion_widget = SuggestionWidget(self)
        self.rephrase_widget = RephraseWidget(self)
        
        # Initialize keyboard monitor with text buffer
        self.keyboard_monitor = KeyboardMonitor(self.text_buffer)
        self.keyboard_monitor.text_captured.connect(self.handle_text_capture)
        self.keyboard_monitor.start()
        
        self.initUI()
        
        # Monitor clipboard for text selection
        self.clipboard = QApplication.clipboard()
        self.clipboard.selectionChanged.connect(self.handle_selection)
        
        # Loading overlay
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.loading_overlay.resize(self.size())
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        central_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 44, 52, 0.95);
                border-radius: 20px;
                border: 1px solid #3d3d3d;
            }
        """)

        # Header
        header = QHBoxLayout()
        
        title = QLabel("✏️ Writing Assistant")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
            }
        """)
        header.addWidget(title)
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ffffff;
                font-size: 20px;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ff4455;
            }
        """)
        header.addWidget(close_btn)
        layout.addLayout(header)

        layout.addSpacing(10)

        # Auto Write button
        auto_write_btn = QPushButton("✨ Auto Write")
        auto_write_btn.setFixedHeight(50)
        auto_write_btn.clicked.connect(self.show_auto_write_dialog)
        auto_write_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d8ce4;
            }
            QPushButton:pressed {
                background-color: #3278c7;
            }
        """)
        layout.addWidget(auto_write_btn)

        # Feature buttons container
        features_container = QWidget()
        features_layout = QHBoxLayout(features_container)
        features_layout.setSpacing(10)

        # Rephrase button
        rephrase_btn = QPushButton("🔄 Rephrase")
        rephrase_btn.setFixedHeight(50)
        rephrase_btn.clicked.connect(self.show_rephrase_dialog)
        rephrase_btn.setStyleSheet("""
            QPushButton {
                background-color: #45a165;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d8956;
            }
            QPushButton:pressed {
                background-color: #357a4b;
            }
        """)
        features_layout.addWidget(rephrase_btn)

        # Complete button
        complete_btn = QPushButton("✨ Complete")
        complete_btn.setFixedHeight(50)
        complete_btn.clicked.connect(self.trigger_completion)
        complete_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #7d3c98;
            }
        """)
        features_layout.addWidget(complete_btn)
        
        layout.addWidget(features_container)

        # Status indicator
        self.status = QLabel("Ready")
        self.status.setStyleSheet("""
            QLabel {
                color: #8f9aab;
                font-size: 14px;
                padding: 10px;
                background-color: rgba(55, 60, 70, 0.95);
                border-radius: 8px;
                min-height: 20px;
            }
        """)
        layout.addWidget(self.status)

        self.setMinimumSize(400, 300)

    def show_auto_write_dialog(self):
        dialog = QWidget(self)
        dialog.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        dialog.setAttribute(Qt.WA_TranslucentBackground)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Styling
        dialog.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 44, 52, 0.95);
                border-radius: 20px;
                border: 1px solid #3d3d3d;
            }
            QTextEdit {
                background-color: rgba(55, 60, 70, 0.95);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                font-size: 14px;
            }
        """)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Auto Write")
        title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        header.addWidget(title)
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(dialog.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ffffff;
                font-size: 20px;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ff4455;
            }
        """)
        header.addWidget(close_btn)
        layout.addLayout(header)
        
        text_input = QTextEdit()
        text_input.setPlaceholderText("Enter what you want to write about...")
        text_input.setMinimumHeight(150)
        layout.addWidget(text_input)
        
        # Loading overlay for dialog
        dialog_loading = LoadingOverlay(dialog)
        dialog_loading.hide()
        
        def generate_with_loading():
            dialog_loading.show()
            self.gen_thread = GenerationThread(self.llm, text_input.toPlainText())
            self.gen_thread.finished.connect(lambda response: self.handle_generated_text(response, dialog, dialog_loading))
            self.gen_thread.start()
        
        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(generate_with_loading)
        generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: #3d8ce4;
            }
        """)
        layout.addWidget(generate_btn)
        
        dialog.setFixedSize(500, 300)
        dialog.move(
            self.x() + (self.width() - dialog.width()) // 2,
            self.y() + (self.height() - dialog.height()) // 2
        )
        
        dialog.show()

    def handle_generated_text(self, response, dialog, loading_overlay):
        pyperclip.copy(response)
        self.keyboard.press(Key.ctrl)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(Key.ctrl)
        loading_overlay.hide()
        dialog.close()
        
    def show_rephrase_dialog(self):
        selected_text = self.clipboard.text(mode=self.clipboard.Selection)
        if selected_text:
            cursor_pos = QCursor.pos()
            self.rephrase_widget.move(cursor_pos.x() + 10, cursor_pos.y() + 10)
            self.rephrase_widget.show()

    def trigger_completion(self):
        buffer_content = self.text_buffer.get()
        if buffer_content:
            self.handle_text_capture(buffer_content)

    @pyqtSlot(str)
    def handle_text_capture(self, text):
        if text.strip():
            prompt = f"""Instructions: You are an autocomplete AI. You will be given text and you need to suggest a natural continuation. Consider the entire context provided. Note: DO NOT PROVIDE ANY TEXT EXCEPT THE CONTINUATION.
Previous text: {text}
Provide a natural continuation:"""
            
            test_prompt="""Instructions: You are an autocomplete AI. You will be given text and you need to suggest a natural continuation. Consider the entire context provided. 
Note: DO NOT PROVIDE ANY TEXT EXCEPT THE CONTINUATION.

Previous text: Hello, I am a very talented Data Analyst and today I am going to
Provide a natural continuation:"""
            
            suggestion = [self.llm(prompt).strip()]
            self.suggestion_widget.suggestions.clear()
            self.suggestion_widget.suggestions.addItems(suggestion)
            
            cursor_pos = QCursor.pos()
            self.suggestion_widget.move(cursor_pos.x() + 10, cursor_pos.y() + 10)
            self.suggestion_widget.show()

    def handle_selection(self):
        selected_text = self.clipboard.text(mode=self.clipboard.Selection)
        if selected_text:
            cursor_pos = QCursor.pos()
            self.rephrase_widget.move(cursor_pos.x() + 10, cursor_pos.y() + 10)
            self.rephrase_widget.show()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def closeEvent(self, event):
        self.keyboard_monitor.stop()
        self.keyboard_monitor.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    assistant = FloatingAssistant()
    assistant.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()