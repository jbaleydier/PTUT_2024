from PIL import Image

def image(i):
    largeur = i.width
    longueur = i.height
    lis = []
    for x in range (1,largeur,1):
        for y in range(1,longueur,1):
            pixel = i.getpixel((x,y))
            lis.append(pixel)
    return(lis)


