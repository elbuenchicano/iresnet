import  numpy   as  np
import  base64
import  re
import  io

from    PIL     import Image
from    faces   import FaceRecognition
from    utils   import u_loadJson

facer   = FaceRecognition()

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
    qr          = []
    students    = u_loadJson('db/list/names.txt')

    for id in students:
        name    = students[id].split('_')
        surname = name[1] if len(name) > 1 else ''

        qr.append( {'id': id,
                    'nombre': name[0],
                    'apellido': surname,
                    'asistencia' : 0} )

    return qr

################################################################################
def updateStudents(data):
    print(data)
    return data

#...............................................................................
################################################################################
################################################################################
def queryImage(data):
    codec   = data[0]['image']
    img     = getI420FromBase64(codec)
    out     = facer.faceRecog(img)
    print(out)
    return  out
