import  torch
import  numpy       as np
import  torch.nn    as nn
import  pandas      as pd
from    torch.utils.data    import DataLoader
from    torchvision         import transforms, datasets
from    facenet_pytorch     import MTCNN, InceptionResnetV1
from    PIL                 import Image
from    utils               import *
from    sklearn.neighbors   import NearestNeighbors
from    joblib              import dump, load




# Define MTCNN module

# Note that, since MTCNN is a collection of neural nets and other code, the
# device must be passed in the following way to enable copying of objects when
# needed internally.


def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product"""
    a = a.cpu()
    b = b.cpu()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def cos(a,b):
    """
    cos_sim returns real numbers,where negative numbers have different interpretations.
    so we use this function to return only positive values.
    """
    minx = -1 
    maxx = 1
    return (cos_sim(a,b)- minx)/(maxx-minx)

def whichDevice():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device

###############################################################################
###############################################################################
###############################################################################
# getting the embeddings for the data set, remember the data set must contain
# names in folders and images inside
def faceRecog(img, embeddings):
    device = whichDevice()
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device
    )
    
    model           = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    x_aligned, prob = mtcnn(x, return_prob=True)
    embedding       = model(aligned)

################################################################################ 
################################################################################ 
################################################################################ 
class FaceRecognition:
    def __init__(self):
        self.device = whichDevice()
        self.mtcnn      = MTCNN(    image_size=160, margin=0, min_face_size=20,
                                    thresholds=[0.6, 0.7, 0.7], 
                                    factor=0.709, device= self.device)
        self.model      = InceptionResnetV1(pretrained='vggface2')
        self.model      = self.model.eval().to(self.device) 
        self.emb_path   = 'db/list'

    #..........................................................................
    # creating embeddings for the dataset
    def getDbEmbeddings(self):
        # remember each folder must coint only pidctures for one person
        dataset = datasets.ImageFolder('db/data/train')
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader  = DataLoader(dataset, collate_fn=lambda x: x[0])

        aligned = []
        names   = []
        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

        aligned     = torch.stack(aligned).to(self.device)
        embeddings  = self.model(aligned)
    
        # saving the names
        emb         = embeddings.cpu().detach().numpy()
        
        u_saveArray2File(self.emb_path+'/names.txt', names)
        u_saveArrayTuple2File(self.emb_path+'/embs.txt', emb)

        nbrs        = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(emb)

        # saving the neares neighbors model
        dump(nbrs, self.emb_path+'/nbrs.joblib')
        
    #..........................................................................
    # face recognition 
    # getting the embeddings for the data set, remember the data set must contain
    # names in folders and images inside
    def faceRecog(self, img):
        nbrs            = load(self.emb_path+'/nbrs.joblib') 
        names           = u_fileList2array(self.emb_path+'/names.txt')
        aligned, prob   = self.mtcnn(img, return_prob=True)
        
        if aligned is not None:
            aligned     = aligned.unsqueeze(0).to(self.device)
            embedding   = self.model(aligned)
            embedding   = embedding.cpu().detach().numpy()                

            distances, indices = nbrs.kneighbors(embedding)

            print(names[indices[0][0]])

        else:
            print('')


        #distances, indices = nbrs.kneighbors(a)

        
        

        