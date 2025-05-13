from .nodes.DetectInnerBox import DetectInnerBox
from .nodes.PasteIntoFrame import PasteIntoFrame
from .nodes.textbox import TextBoxAutoWrap

NODE_CLASS_MAPPINGS = {
    "DetectInnerBox": DetectInnerBox,
    "PasteIntoFrame": PasteIntoFrame,
    "TextBoxAutoWrap": TextBoxAutoWrap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectInnerBox": "Detect Inner Box",
    "PasteIntoFrame": "Paste Into Frame",
    "TextBoxAutoWrap": "Text Box Auto Wrap",
}
