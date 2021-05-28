import  torch
import  numpy       as np
import  torch.nn    as nn
from    torch.utils.data    import DataLoader
from    torchvision         import transforms, datasets
from    facenet_pytorch     import MTCNN, InceptionResnetV1
from    PIL                 import Image
from    utils               import *
from    sklearn.neighbors   import NearestNeighbors
from    joblib              import dump, load


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

        try:
            self.nbrs       = load(self.emb_path+'/nbrs.joblib') 
            self.targets    = u_fileList2array(self.emb_path+'/targets.txt')
            self.names      = u_loadJson(self.emb_path+'/names.txt')

        except:
            print('First define the embeddings' +
                  'running getDbEmbeddings function')

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

        aligned     = torch.stack(aligned).to(self.device)
        embeddings  = self.model(aligned)
        emb         = embeddings.cpu().detach().numpy()
        nbrs        = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(emb)

        # saving the neares neighbors model and other files
        dump(nbrs, self.emb_path+'/nbrs.joblib')
        u_saveArray2File(self.emb_path+'/targets.txt', dataset.targets)
        u_saveDict2File(self.emb_path+'/names.txt', dataset.idx_to_class)
        u_saveArrayTuple2File(self.emb_path+'/embs.txt', emb)
        
    #..........................................................................
    # face recognition 
    # getting the embeddings for the data set, remember the data set must contain
    # names in folders and images inside
    def faceRecog(self, img):
        aligned, prob   = self.mtcnn(img, return_prob=True)
        data            = {'id': -1, 'dist': -1}
       
        if aligned is not None:
            aligned     = aligned.unsqueeze(0).to(self.device)
            embedding   = self.model(aligned)
            embedding   = embedding.cpu().detach().numpy()                

            distances, indices = self.nbrs.kneighbors(embedding)
            
            if len(distances) > 0: 
                data['id']      = int(self.targets[indices[0][0]]) 
                data['dist']    = distances[0][0]   - 0.3 

        return [data]  

        
        

        
