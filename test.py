''' importing the nst module '''
from nst import NST
import matplotlib.pyplot as plt

''' initializing the NST Model '''
nst_res = NST(gpu=True)

''' reading the content and style image path '''
content_img, style_img = ('<path to content image>', '<path to style image')

''' running inferences '''
result = nst_res.run_nst(content_img, style_img)

''' displaying the resulting image '''
plt.imshow(result)
plt.show()