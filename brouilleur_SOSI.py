################################# 07/09/16 ####################################
# Ce programme est un brouilleur d'ecran. Il empeche des personnes tierces    #
# de regarder l'ecran de l'utilisateur a son insu. Lorsqu'un visage inconnu   #
# est detecte par la webcam, l'image filmee est affichee en plein ecran.      #
# Ainsi, l'utilisateur est informe de la presence d'un intrus, il peut alors  #
# soit desactiver completement le programme, soit autoriser l'intrus a        #
# regarder l'ecran.                                                           #
###############################################################################

#### Auteurs ######
# Gautier DARCHEN #
# Enora GICQUEL   #
# Alexandre HUAT  #
# Romain JUDIC    #
###################


import cv2
import sys

####################
#    Constantes    #
####################

GLANDE_OPT = "-glande"
IRON_MAN_OPT = "-iron"

####################
#    Fonctions     #
####################

def plusGrandVisage(faces):
	"""Cette fonction retourne la surface du plus grand visage parmi ceux detectes."""
	maxi = 0
	for (x,y,w,h) in faces:
		if w*h>maxi:
			maxi = w*h
	return maxi

def afficherBrouilleur(frame,option):
        cv2.namedWindow('Alert', cv2.WND_PROP_FULLSCREEN)          
        cv2.setWindowProperty('Alert', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if option == GLANDE_OPT:
                img = cv2.imread('cheat.png',cv2.IMREAD_COLOR )
                cv2.imshow('Alert', img)
        else:
                """Cette fonction affiche le brouilleur, incluant la video filmee par la webcam (detectant les visages), et affichant les instructions pour l'utilisateur."""
                height, width = frame.shape[:2]

                cv2.namedWindow('Alert', cv2.WND_PROP_FULLSCREEN)          
                cv2.setWindowProperty('Alert', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                frame = cv2.flip(frame, 1)
                cv2.putText(frame,"Intruder has been detected !",(100,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,250), 2)
                cv2.putText(frame,"Press Space to add a friend.",(125,height-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (25,25,25), 2)
                cv2.putText(frame,"Hold ESC to quit.",(175,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (25,25,25), 2)
                cv2.imshow('Alert', frame)
                

def ajouterRectangles(faces,frame,nbFrames):
	"""Cette fonction dessine des rectangles autour des visages detectes dans le tableau 'faces'. Elle permet de limiter la reconnaissance de visages "parasites" en considerant qu'il faut qu'une detection apparaisse sur 100 frames successives pour etre considere comme un visage."""
        for (x, y, w, h) in faces:
                if not (w*h<plusGrandVisage(faces) & nbFrames < 100):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def ajouterMasques(faces,frame,nbFrames,image,mask,maskInv):
        """Cette fonction dessine des masques autour des visages detectes dans le tableau 'faces'. Elle permet de limiter la reconnaissance de visages "parasites" en considerant qu'il faut qu'une detection apparaisse sur 100 frames successives pour etre considere comme un visage."""
        for (x, y, w, h) in faces:
                if not (w*h<plusGrandVisage(faces) & nbFrames < 100):
                        # cree une image cadree sur le visage
                        imgFace = frame[y:y+h, x:x+w]
                        # redimensionne l'image et les masques a la taille du visage
                        image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
                        mask = cv2.resize(mask, (w,h), interpolation = cv2.INTER_AREA)
                        maskInv = cv2.resize(maskInv, (w,h), interpolation = cv2.INTER_AREA)
                        # cree un premier plan contenant la partie en couleur de l'image
                        background = cv2.bitwise_and(imgFace,imgFace,mask = maskInv)
                        # cree un fond contenant l'image du visage sur la partie en transparence de l'image
                        foreground = cv2.bitwise_and(image,image,mask = mask)
                        # cree une image en superposant le fond et le premier plan
                        visageMasque = cv2.add(background,foreground)
                        # remplace le visage par l'image creee sur la frame
                        frame[y:y+h, x:x+w] = visageMasque

def creerMasques(imagePath):
        # recupere l'image avec la transparence
        img = cv2.imread(imagePath,-1)
        # cree un masque representant la partie en couleur de l'image
        mask = img[:,:,3]
		# cree un masque representant la partie en transparence de l'image
        maskInv = cv2.bitwise_not(mask)
		# l'image passe en couleur sans transparence
        img = img[:,:,0:3]
        return img, mask, maskInv

#######################
# Programme principal #
#######################

option = ""
if len(sys.argv) > 1:
        option = sys.argv[1]
     
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

if option == IRON_MAN_OPT:
        imgIronMan, maskIronMan, maskIronManInv = creerMasques('ironman.png')

video_capture = cv2.VideoCapture(0)

nbFrames = 0
nbVisages = 0
nbVisagesAutorises = 1
continuerAffichage = False


while True:
        # Capture frame par frame
        ret, frame = video_capture.read()
	
	# On passe l'image en niveaux de gris pour que les traitements soient plus rapides
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detection de visages dans 'faces' sous forme de tableau (x,y,largeur,hauteur)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

	# Des qu'un nouveau visage apparait, on compte le nombre de frames successives sur lequel on le trouve pour filtrer les parasites
        if len(faces)>=nbVisages:
                nbVisages = len(faces)
                if nbVisages>1:
                        nbFrames+=1
        else:
                nbVisages = len(faces)
                nbFrames = 0

	# On brouille l'ecran quand il y a plus de visages detectes que le nombre autorise
        if nbVisages > nbVisagesAutorises:
                if option == IRON_MAN_OPT:
                        ajouterMasques(faces,frame,nbFrames,imgIronMan,maskIronMan,maskIronManInv)
                else:
                        ajouterRectangles(faces,frame,nbFrames)       
                afficherBrouilleur(frame,option)
                continuerAffichage = True
		# Si l'utilisateur appuie sur ESPACE (32) on autorise la nouvelle personne detectee et on masque le brouilleur
                if cv2.waitKey(1) & 0xFF == 32:
                        continuerAffichage = False
                        cv2.destroyAllWindows()
                        nbVisagesAutorises=nbVisages
        else:
		# Cas ou le visage en trop est parti mais que l'utilisateur n'a pas desactive le brouilleur
                if continuerAffichage:
                        if option == IRON_MAN_OPT:
                                ajouterMasques(faces,frame,nbFrames,imgIronMan,maskIronMan,maskIronManInv)
                        else:
                                ajouterRectangles(faces,frame,nbFrames) 
                        afficherBrouilleur(frame,option) 
		
		# Mise a jour du nombre de visages autorises lorsqu'une personne quitte le champ de vision de la webcam
                if nbVisages > 1:
                        nbVisagesAutorises=nbVisages
                else:
                        nbVisagesAutorises=1
			# S'il ne reste qu'une personne, un appui sur ESPACE doit aussi masquer le brouilleur
                        if cv2.waitKey(1) & 0xFF == 32:
                                continuerAffichage = False
                                cv2.destroyAllWindows()
                                nbVisagesAutorises=nbVisages
                        
        # Si l'utilisateur appuie sur la touche ESC (27) on quitte le programme
        if cv2.waitKey(1) & 0xFF == 27:
                continuerAffichage = False
                break

# On eteint la webcam et on ferme les fenetres
video_capture.release()
cv2.destroyAllWindows()
