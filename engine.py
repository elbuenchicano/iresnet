import numpy as np
import PIL

import  base64
import  re
import  io

from    PIL     import Image

################################################################################
################################################################################
def getI420FromBase64(codec):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data   = base64.b64decode(base64_data)
    image_data  = io.BytesIO(byte_data)
    img         = Image.open(image_data)
    return img 

    #img.save(image_path + 'img_' + str(ticket) +'.png', "PNG")
    #ticket += 1

################################################################################
################################################################################
# esta parte va ser cambiada pero recien en producción por el momento solo 
# personas de forma manual
################################################################################
################################################################################
def queryStudents():
    qr = []
    for i in range(5):
        qr.append( {'id': i, 'nombre': 'nombre_'+str(i), 
                    'apellido': 'apellido_'+str(i), 'asistencia' : 0})

    return qr
################################################################################
def updateStudents(data):
    print(data)
    return data

#...............................................................................
################################################################################
################################################################################
def queryImage(data):
    codec = data[0]['image']
    print(codec)
    return {'a': 2}
