from .DetectInnerBox import DetectInnerBox
from .PasteIntoFrame import PasteIntoFrame

NODE_CLASS_MAPPINGS = {
    "DetectInnerBox": DetectInnerBox,
    "PasteIntoFrame": PasteIntoFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectInnerBox": "Detect Inner Box",
    "PasteIntoFrame": "Paste Into Frame",
}
