import numpy as np
import io
from PIL import Image
import matplotlib


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    from io import BytesIO

    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent=True)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.asarray(Image.open(BytesIO(img_arr)))
    buf.close()
    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img
