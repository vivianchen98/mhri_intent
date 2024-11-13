import numpy as np
from PIL import Image

XMIN_PIX = 80
YMIN_PIX = 80
XMAX_PIX = 642
YMAX_PIX = 642

DETECTION_SIZE = 16


class CanvasParser:
    def __init__(self, detection_size=DETECTION_SIZE):
        x_locs = np.linspace(XMIN_PIX, XMAX_PIX, num=19)
        y_locs = np.linspace(YMIN_PIX, YMAX_PIX, num=19)
        self.mask = np.zeros((19 * 19, 720, 720), dtype=bool)
        for i, x in enumerate(x_locs):
            for j, y in enumerate(y_locs):
                self.mask[
                    i * 19 + j,
                    int(x - detection_size // 2) : int(x + detection_size // 2),
                    int(y - detection_size // 2) : int(y + detection_size // 2),
                ] = True

    def parse_as_mask(self, canvas: Image.Image) -> np.ndarray:
        if canvas.size != (720, 720):
            canvas = canvas.resize((720, 720))
        canvas_array = np.array(canvas)[:, :, 3] > 0
        masked_canvas = np.logical_and(self.mask, canvas_array[None])
        return masked_canvas.any(axis=(1, 2)).reshape(19, 19)

    def parse_as_text(self, canvas: Image.Image) -> str:
        mask = self.parse_as_mask(canvas)
        result_string = ""
        for i in range(19):
            for j in range(19):
                if mask[i, j]:
                    result_string += "1"
                else:
                    result_string += "0"
            if i != 18:
                result_string += "\n"
        return result_string

    def get_mask_from_grid(self, x: int, y: int) -> np.ndarray:
        return self.mask[(2 * x + 1) * 19 + 2 * y + 1]


if __name__ == "__main__":
    canvas_parser = CanvasParser()
    canvas = Image.open("/Users/rzhao/research/multimodal-hri-dev/web_ui_trajectories/p0_2.png")
    print(canvas_parser.parse_as_text(canvas))
