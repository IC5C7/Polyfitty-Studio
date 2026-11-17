from __future__ import annotations

"""Polyfitty Studio brand themes and Windows system-theme resolution."""

import sys

THEMES = {
    "graphite": {
        "label": "Studio Dark",
        "window": "#080D19", "base": "#0C1423", "panel": "#111D31",
        "panel_alt": "#182941", "border": "#29415F", "text": "#EAF2FF",
        "muted": "#91A8C6", "accent": "#32B8FF", "accent_alt": "#B18BFF", "accent_hover": "#69CCFF", "accent_text": "#06111E",
        "button": "#1A304B", "button_hover": "#254463", "selection": "#236CA6",
        "success": "#53D6B0", "success_bg": "#123C3A", "warning": "#FFC56E",
        "warning_bg": "#4A351C", "error": "#FF858F", "error_bg": "#4B2531",
        "info": "#73C9FF", "info_bg": "#163B59", "scene": (8, 17, 32, 255),
    },
    "midnight": {
        "label": "Deep Space",
        "window": "#08091A", "base": "#0D1025", "panel": "#151A35",
        "panel_alt": "#20264A", "border": "#35416E", "text": "#EEF0FF",
        "muted": "#A1A9D0", "accent": "#8F83FF", "accent_alt": "#43D7E8", "accent_hover": "#B0A8FF", "accent_text": "#0C0B23",
        "button": "#272F59", "button_hover": "#354273", "selection": "#514BA4",
        "success": "#62DCC0", "success_bg": "#153E43", "warning": "#FFD17B",
        "warning_bg": "#4B3A28", "error": "#FF8DAB", "error_bg": "#51283D",
        "info": "#91BFFF", "info_bg": "#242F63", "scene": (12, 15, 39, 255),
    },
    "ocean": {
        "label": "Arctic Cyan",
        "window": "#061622", "base": "#092131", "panel": "#103044",
        "panel_alt": "#164057", "border": "#286078", "text": "#E4FAFF",
        "muted": "#91C1D1", "accent": "#35D5E8", "accent_alt": "#B7A2FF", "accent_hover": "#73EAF5", "accent_text": "#05202A",
        "button": "#195064", "button_hover": "#21677C", "selection": "#187D98",
        "success": "#68E0BE", "success_bg": "#144C4A", "warning": "#FFD17B",
        "warning_bg": "#514022", "error": "#FF929B", "error_bg": "#542C3A",
        "info": "#7AD8F5", "info_bg": "#16465F", "scene": (6, 26, 39, 255),
    },
    "light": {
        "label": "Studio Light",
        "window": "#E8EFF8", "base": "#F3F7FC", "panel": "#FFFFFF",
        "panel_alt": "#EAF1FA", "border": "#C8D8EA", "text": "#172B45",
        "muted": "#60758F", "accent": "#0C73B8", "accent_alt": "#7650C8", "accent_hover": "#095F9F", "accent_text": "#FFFFFF",
        "button": "#E3EDF8", "button_hover": "#D2E4F5", "selection": "#B8DDF7",
        "success": "#087F68", "success_bg": "#D9F3EB", "warning": "#94600A",
        "warning_bg": "#FFF0D1", "error": "#B4233A", "error_bg": "#FCE5E9",
        "info": "#096FAF", "info_bg": "#DCEFFC", "scene": (229, 237, 248, 255),
    },
    "ember": {
        "label": "Violet Pulse",
        "window": "#120D22", "base": "#19132E", "panel": "#241B3F",
        "panel_alt": "#30244F", "border": "#584477", "text": "#F3EDFF",
        "muted": "#B9A9D7", "accent": "#B18BFF", "accent_alt": "#45CBEA", "accent_hover": "#C9ADFF", "accent_text": "#170F29",
        "button": "#3A2B5C", "button_hover": "#4B3675", "selection": "#7553B8",
        "success": "#6EDBC3", "success_bg": "#16463F", "warning": "#FFD17B",
        "warning_bg": "#51402A", "error": "#FF91A5", "error_bg": "#562C43",
        "info": "#A5B9FF", "info_bg": "#322B68", "scene": (25, 17, 45, 255),
    },
}

def detect_system_theme() -> str:
    """Return the Windows app-mode theme, falling back safely to Studio Dark."""
    if sys.platform != "win32":
        return "graphite"

    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path)
        try:
            for value_name in ("AppsUseLightTheme", "SystemUsesLightTheme"):
                try:
                    value, _value_type = winreg.QueryValueEx(key, value_name)
                except OSError:
                    continue
                return "light" if int(value) == 1 else "graphite"
        finally:
            winreg.CloseKey(key)
    except (ImportError, OSError, TypeError, ValueError, AttributeError):
        pass

    return "graphite"

def resolve_theme_name(theme_mode: str, system_theme: str | None = None) -> str:
    """Resolve a persisted theme mode to one of the concrete theme palettes."""
    mode = str(theme_mode or "system").strip().lower()
    if mode == "system":
        resolved = str(system_theme or detect_system_theme()).strip().lower()
        return resolved if resolved in THEMES else "graphite"
    return mode if mode in THEMES else "graphite"

def initial_theme_mode(saved_theme_mode: str | None, legacy_theme: str | None = None) -> str:
    """Choose the persisted user mode, defaulting clean installs to System."""
    if saved_theme_mode is not None:
        saved = str(saved_theme_mode).strip().lower()
        return saved if saved == "system" or saved in THEMES else "system"
    legacy = str(legacy_theme or "").strip().lower()
    if legacy in THEMES and legacy != "graphite":
        return legacy
    return "system"

def stylesheet_for(theme_name: str) -> str:
    theme = THEMES.get(str(theme_name).lower(), THEMES["graphite"])
    return f"""
QWidget {{ background: {theme['base']}; color: {theme['text']}; font-family: 'Segoe UI'; font-size: 10pt; }}
QMainWindow {{ background: {theme['window']}; }}
QFrame#Header {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {theme['panel']}, stop:1 {theme['panel_alt']}); border: 1px solid {theme['border']}; border-radius: 14px; }}
QFrame#Panel {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 12px; }}
QFrame#Metric {{ background: {theme['panel_alt']}; border: 1px solid {theme['border']}; border-radius: 10px; }}
QLabel#Brand {{ color: {theme['text']}; font-size: 19pt; font-weight: 800; letter-spacing: 2px; }}
QLabel#SubTitle {{ color: {theme['muted']}; font-size: 9pt; }}
QLabel#Section {{ color: {theme['accent']}; font-size: 10pt; font-weight: 700; }}
QLabel#MetricValue {{ color: {theme['text']}; font-size: 15pt; font-weight: 700; }}
QLabel#MetricName {{ color: {theme['muted']}; font-size: 8pt; font-weight: 600; }}
QLabel#ShortcutKey {{ color: {theme['accent']}; background: {theme['panel_alt']}; border: 1px solid {theme['border']}; border-radius: 4px; padding: 2px 6px; font-weight: 800; }}
QLabel#ProgressPercent {{ color: {theme['accent']}; font-weight: 800; font-size: 10pt; }}
QFrame#DiagnosticsPanel QLabel#Section {{ color: {theme['text']}; font-size: 9pt; font-weight: 750; }}
QFrame#DiagnosticsPanel QLabel#MetricValue,
QFrame#DiagnosticsPanel QLabel#DiagnosticsMetricValue {{ color: {theme['text']}; font-size: 8pt; font-weight: 650; }}
QFrame#DiagnosticsPanel QLabel#MetricName,
QFrame#DiagnosticsPanel QLabel#Muted {{ color: {theme['muted']}; }}
QLabel#GPUStatus {{ background: {theme['panel_alt']}; color: {theme['muted']}; border: 1px solid {theme['border']}; border-radius: 7px; padding: 6px 8px; font-size: 8pt; font-weight: 800; }}
QLabel#GPUStatus[state="detected"] {{ color: {theme['text']}; border-color: {theme['accent']}; }}
QLabel#GPUStatus[state="ready"] {{ color: {theme['info']}; background: {theme['info_bg']}; border-color: {theme['info']}; }}
QLabel#GPUStatus[state="active"] {{ color: {theme['success']}; background: {theme['success_bg']}; border-color: {theme['success']}; }}
QLabel#GPUStatus[state="warning"], QLabel#GPUStatus[state="cpu"] {{ color: {theme['warning']}; background: {theme['warning_bg']}; border-color: {theme['warning']}; }}
QLabel#GPUStatus[state="error"] {{ color: {theme['error']}; background: {theme['error_bg']}; border-color: {theme['error']}; }}
QFrame#DiagnosticsPanel QTextEdit#DiagnosticsConsole {{ background: {theme['window']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 8px; font-family: Consolas; font-size: 8pt; padding: 3px; }}
QLabel#SceneTitle {{ color: {theme['text']}; font-size: 10pt; font-weight: 700; }}
QLabel#Muted {{ color: {theme['muted']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {theme['window']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 7px; padding: 7px 8px; selection-background-color: {theme['selection']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {theme['accent']}; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QPushButton {{ background: {theme['button']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 8px; padding: 9px 12px; font-weight: 650; }}
QPushButton:hover {{ background: {theme['button_hover']}; border-color: {theme['accent']}; }}
QPushButton:checked {{ background: {theme['selection']}; border-color: {theme['accent']}; color: {theme['text']}; }}
QPushButton:disabled {{ color: {theme['muted']}; background: {theme['panel_alt']}; border-color: {theme['border']}; }}
QPushButton#Primary {{ background: {theme['accent']}; border: 1px solid {theme['accent_hover']}; color: {theme['accent_text']}; font-size: 11pt; font-weight: 800; padding: 12px; }}
QPushButton#Primary:hover {{ background: {theme['accent_hover']}; }}
QPushButton#Export {{ background: {theme['success_bg']}; border: 1px solid {theme['success']}; color: {theme['success']}; }}
QToolButton#DiagnosticsToggle {{ background: {theme['panel_alt']}; border: 1px solid {theme['border']}; border-radius: 8px; padding: 7px 10px; color: {theme['text']}; font-weight: 750; text-align: left; }}
QToolButton#DiagnosticsToggle:hover {{ border-color: {theme['accent']}; }}
QToolButton#DiagnosticsToggle:checked {{ color: {theme['text']}; border-color: {theme['accent']}; }}
QToolButton#CollapsibleHeader {{ background: {theme['panel_alt']}; border: 1px solid {theme['border']}; border-radius: 8px; padding: 7px 10px; color: {theme['accent']}; font-weight: 750; text-align: left; }}
QToolButton#CollapsibleHeader:hover {{ border-color: {theme['accent']}; }}
QToolButton#CollapsibleHeader:checked {{ color: {theme['text']}; border-color: {theme['accent']}; }}
QGroupBox {{ border: 1px solid {theme['border']}; border-radius: 9px; margin-top: 12px; padding: 10px 9px 8px 9px; font-weight: 700; color: {theme['accent']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
QProgressBar {{ background: {theme['window']}; border: 1px solid {theme['border']}; border-radius: 7px; height: 16px; text-align: center; color: {theme['text']}; font-size: 8pt; }}
QProgressBar::chunk {{ background: {theme['accent']}; border-radius: 6px; }}
QTextEdit {{ background: {theme['window']}; border: 1px solid {theme['border']}; border-radius: 8px; color: {theme['text']}; font-family: Consolas; font-size: 9pt; padding: 6px; }}
QTableWidget, QTreeWidget, QListWidget {{ background: {theme['window']}; alternate-background-color: {theme['panel']}; color: {theme['text']}; border: 1px solid {theme['border']}; gridline-color: {theme['border']}; selection-background-color: {theme['selection']}; }}
QHeaderView::section {{ background: {theme['panel_alt']}; color: {theme['text']}; border: 0; border-bottom: 1px solid {theme['border']}; padding: 6px; font-weight: 700; }}
QTabWidget::pane {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 8px; }}
QTabBar::tab {{ background: {theme['panel_alt']}; color: {theme['muted']}; border: 1px solid {theme['border']}; padding: 7px 12px; margin-right: 3px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {theme['panel']}; color: {theme['accent']}; border-bottom-color: {theme['panel']}; }}
QMenu {{ background: {theme['panel']}; color: {theme['text']}; border: 1px solid {theme['border']}; padding: 4px; }}
QMenu::item:selected {{ background: {theme['selection']}; color: {theme['text']}; }}
QToolTip {{ background: {theme['panel_alt']}; color: {theme['text']}; border: 1px solid {theme['border']}; padding: 4px 6px; }}
QSplitter::handle {{ background: {theme['panel_alt']}; }}
QScrollArea {{ background: transparent; border: 0; }}
QScrollBar:vertical {{ background: {theme['base']}; width: 13px; margin: 2px 1px 2px 1px; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {theme['muted']}; min-height: 30px; border: 1px solid {theme['border']}; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {theme['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: {theme['base']}; height: 13px; margin: 1px 2px 1px 2px; border-radius: 6px; }}
QScrollBar::handle:horizontal {{ background: {theme['muted']}; min-width: 30px; border: 1px solid {theme['border']}; border-radius: 5px; }}
QScrollBar::handle:horizontal:hover {{ background: {theme['accent']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
"""
