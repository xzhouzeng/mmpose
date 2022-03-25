
from PIL import Image
import numpy as np
 
I = Image.open("hpe-picture/test-p2.jpg") 
I.show()    
# I.save('./save.png')
I_array = np.array(I)
print(I_array.shape)
# mg2 = Image.fromarray(mats)